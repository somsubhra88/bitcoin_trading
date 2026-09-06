"""Paper-trading adapter: safety, idempotency, and parity with the training env.

The claim this module exists to defend is narrow and load-bearing: **the paper
broker reproduces :class:`TradingEnv` exactly**. If it does not, then every
backtest number describes a simulator that the deployment path does not implement,
and the walk-forward report is measuring the wrong program.

"Exactly" is meant literally — :func:`test_broker_reproduces_the_training_env`
requires a maximum absolute equity deviation of ``0.0``, not merely a small one. A
tolerance there would hide precisely the bugs worth finding: a fee applied at a
different mark, a fill priced off the close instead of the open, a latency queue
drained one bar early.

Two parity claims are checked separately, because the risk shield sits between
them:

*model* parity  the decoder-plus-adapter action, pre-shield, must match the
                training path — i.e. the deployed agent computes the same decision.
*executed* parity  the post-shield, post-latency target must match — i.e. the
                runner's four gates (agent, shield, operator limits, latency) are
                ordered the way the environment orders them.

Conflating them produces exactly as many false mismatches as there were shield
interventions, which is how this was originally mis-diagnosed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.archetype.conftest import DEVICE


@pytest.fixture(scope="module")
def paper_result(bundle, stack, tmp_path_factory):
    """One full paper run over the test segment, shared by the parity checks."""
    from archetype_trader.utils.seeding import seed_everything

    journal = tmp_path_factory.mktemp("journal") / "paper.jsonl"
    seed_everything(0)
    runner = _make_runner(bundle, stack, run_id="paper-test", journal=journal)
    return runner.run()


def _seg_start(bundle, segment) -> int:
    """First bar on which a decision can be taken: features warm and context full."""
    cfg, features = bundle.cfg, bundle.features
    return max(
        segment.start, features.valid_from, features.lookback - 1, cfg.phase2.context_bars - 1
    )


def _make_runner(
    bundle,
    stack,
    *,
    run_id: str,
    journal=None,
    fault=None,
    limits=None,
    with_phase3: bool = True,
    shield_cfg=None,
):
    from archetype_trader.live import PaperBroker, PaperRunner, build_agent
    from archetype_trader.live.agent import LoadedStack

    cfg, segment = bundle.cfg, bundle.fold.test
    start = _seg_start(bundle, segment)
    ecfg = cfg.env.model_copy(update={"max_episode_bars": max(2, segment.stop - start)})
    broker = PaperBroker(
        bundle.data,
        ecfg,
        start_index=start,
        end_index=segment.stop,
        symbol=cfg.data.symbol,
        bars_per_year=cfg.eval.bars_per_year,
        vol_lookback=cfg.shield.vol_lookback,
        fault_injector=fault,
    )
    loaded = LoadedStack(
        model=stack.phase1.model,
        policy=None if stack.phase2 is None else stack.phase2.policy,
        adapter=None if (stack.phase3 is None or not with_phase3) else stack.phase3.model,
        positions=cfg.env.positions,
        horizon=cfg.phase1.horizon,
        context_bars=cfg.phase2.context_bars,
        include_cash_archetype=cfg.phase2.include_cash_archetype,
        n_archetypes=stack.n_archetypes,
        override_budget=cfg.phase3.override_budget,
        shadow_cost_frac=0.0 if stack.dataset is None else stack.dataset.cost_frac,
    )
    return PaperRunner(
        build_agent(loaded, device=DEVICE),
        broker,
        bundle.data,
        stack.scaled,
        ecfg,
        shield_cfg if shield_cfg is not None else cfg.shield,
        run_id=run_id,
        bars_per_year=cfg.eval.bars_per_year,
        horizon_hint=cfg.phase1.horizon,
        journal_path=journal,
        limits=limits,
        # No wall-clock sleeping in tests; the runner's pacing is not under test.
        sleep=lambda _s: None,
    )


# --------------------------------------------------------------------------- #
# 1. Safety pre-flight
# --------------------------------------------------------------------------- #
def test_paper_is_approved_and_live_is_refused(cfg) -> None:
    from archetype_trader.live import preflight

    paper = preflight(cfg, mode="paper")
    assert paper.approved, paper.blockers

    live = preflight(cfg, mode="live")
    assert not live.approved
    assert any("--enable-live" in b for b in live.blockers), live.blockers


def test_live_stays_refused_even_when_fully_armed(cfg) -> None:
    """Every operator gate satisfied, and live is *still* refused.

    This is the safety property the README claims: refusal does not depend on the
    operator forgetting a flag. There is no exchange adapter in the repository, so
    there is no code path that can place a real order, and adding one is a commit
    that must also carry its own tests.
    """
    from archetype_trader.live import LiveLimits, preflight

    os.environ["ARCHETYPE_LIVE_API_KEY"] = "dummy-key-for-the-test"
    os.environ["ARCHETYPE_LIVE_API_SECRET"] = "dummy-secret-for-the-test"
    try:
        armed = preflight(
            cfg,
            mode="live",
            enable_live=True,
            exchange="binance",
            testnet=True,
            confirm_phrase="I ACCEPT THE RISK",
            limits=LiveLimits(
                max_order_notional=100.0,
                max_position_notional=500.0,
                max_leverage=2.0,
                require_testnet=True,
            ),
        )
    finally:
        os.environ.pop("ARCHETYPE_LIVE_API_KEY", None)
        os.environ.pop("ARCHETYPE_LIVE_API_SECRET", None)

    assert not armed.approved
    assert any("no live adapter exists" in b for b in armed.blockers), armed.blockers


def test_disabling_the_shield_warns_on_paper_and_blocks_live(cfg) -> None:
    from archetype_trader.live import preflight

    unshielded = cfg.model_copy(update={"shield": cfg.shield.model_copy(update={"enabled": False})})
    p = preflight(unshielded, mode="paper")
    assert p.approved and any("risk shield is disabled" in w for w in p.warnings), p.warnings

    live = preflight(unshielded, mode="live", enable_live=True)
    assert any("without the deterministic shield is refused" in b for b in live.blockers), (
        live.blockers
    )


# --------------------------------------------------------------------------- #
# 2. Secrets
# --------------------------------------------------------------------------- #
def test_secret_scan_finds_key_shaped_and_value_shaped_leaks() -> None:
    from archetype_trader.live import scan_for_secrets

    assert scan_for_secrets({"data": {"path": "x.csv"}, "env": {"mode": "perp"}}) == []

    by_key = scan_for_secrets({"exchange": {"api_key": "abc"}, "nested": [{"secret": "s"}]})
    assert set(by_key) == {"exchange.api_key", "nested[0].secret"}, by_key

    # Caught by shape, not by name: a long opaque blob under an innocent key.
    assert scan_for_secrets({"note": "A" * 40}) == ["note"]


def test_credentials_never_appear_in_a_repr() -> None:
    """A traceback or a debug log must not be able to print a secret."""
    from archetype_trader.live.safety import Credentials

    c = Credentials(key="abcdefghijklmnop", secret="topsecret")
    assert "topsecret" not in repr(c)
    assert "topsecret" not in str(c)


# --------------------------------------------------------------------------- #
# 3. Idempotency: the two distinct transport-failure windows
# --------------------------------------------------------------------------- #
def _broker(bundle, *, fault=None, ack_fault=None, offset: int = 100):
    from archetype_trader.live import PaperBroker

    cfg = bundle.cfg
    ecfg = cfg.env.model_copy(update={"max_episode_bars": 40})
    start = bundle.fold.train.start + offset
    return PaperBroker(
        bundle.data,
        ecfg,
        start_index=start,
        end_index=start + 40,
        symbol="S",
        fault_injector=fault,
        ack_fault_injector=ack_fault,
    )


def test_duplicate_submission_is_acknowledged_once_and_fills_once(bundle) -> None:
    from archetype_trader.live import client_order_id
    from archetype_trader.live.broker import Order

    b = _broker(bundle)
    oid = client_order_id(run_id="t", index=b.index, target_position=1.0)
    order = Order(client_order_id=oid, symbol="S", target_position=1.0, decided_at=b.index)

    first = b.submit(order)
    second = b.submit(order)
    assert not first.duplicate
    assert second.duplicate
    assert len(b.open_orders()) == 1, b.open_orders()
    assert len(b.advance().fills) == 1


def test_failure_before_the_broker_records_the_order_is_safe_to_retry(bundle) -> None:
    """Window one: the request never landed. Retrying must create exactly one order."""
    from archetype_trader.live import FlakyTransport, client_order_id
    from archetype_trader.live.broker import BrokerError, Order

    b = _broker(bundle, fault=FlakyTransport(n_failures=2))
    order = Order(
        client_order_id=client_order_id(run_id="t", index=b.index, target_position=1.0),
        symbol="S",
        target_position=1.0,
        decided_at=b.index,
    )
    raised = 0
    ack = None
    for _ in range(3):
        try:
            ack = b.submit(order)
        except BrokerError:
            raised += 1
            continue
        break
    assert raised == 2
    assert ack is not None and not ack.duplicate
    assert len(b.open_orders()) == 1


def test_a_lost_acknowledgement_does_not_double_the_position(bundle) -> None:
    """Window two: the broker HAS the order but the caller never heard back.

    This is the dangerous one. A blind retry here is the classic route to double
    exposure, and the ``client_order_id`` idempotency key is the only thing that
    prevents it.
    """
    from archetype_trader.live import FlakyTransport, client_order_id
    from archetype_trader.live.broker import BrokerError, Order

    b = _broker(bundle, ack_fault=FlakyTransport(n_failures=1))
    order = Order(
        client_order_id=client_order_id(run_id="t", index=b.index, target_position=1.0),
        symbol="S",
        target_position=1.0,
        decided_at=b.index,
    )
    with pytest.raises(BrokerError):
        b.submit(order)

    retry = b.submit(order)  # blind retry, caller does not know the first landed
    assert retry.duplicate, "the idempotency key failed to catch a lost-ack retry"
    assert len(b.open_orders()) == 1, b.open_orders()
    assert len(b.advance().fills) == 1, "the position doubled"


def test_client_order_ids_are_deterministic_but_distinct_per_attempt() -> None:
    from archetype_trader.live import client_order_id

    a = client_order_id(run_id="t", index=7, target_position=1.0)
    assert a == client_order_id(run_id="t", index=7, target_position=1.0)
    assert a != client_order_id(run_id="t", index=7, target_position=1.0, attempt=1)
    assert a != client_order_id(run_id="t", index=8, target_position=1.0)
    assert a != client_order_id(run_id="t", index=7, target_position=0.5)
    assert a != client_order_id(run_id="other", index=7, target_position=1.0)


# --------------------------------------------------------------------------- #
# 4. The runner end to end
# --------------------------------------------------------------------------- #
def test_runner_submits_one_order_per_bar_and_journals_every_gate(paper_result) -> None:
    res = paper_result
    m = res.metrics
    assert res.bars > 0
    assert m.orders_submitted == res.bars

    assert res.journal_path is not None and res.journal_path.exists()
    n_lines = sum(1 for _ in res.journal_path.open())
    # One line per bar, plus a session-start and a session-end record.
    assert n_lines == res.bars + 2, f"{n_lines} journal lines for {res.bars} bars"

    # All four deployment gates must be journalled separately, or an intervention
    # is indistinguishable from a decision.
    for key in (
        "proposed_position",
        "shielded_position",
        "limited_position",
        "executed_target",
        "reconciled",
        "position_drift",
    ):
        assert key in res.decisions[0], key


def test_paper_run_reconciles(paper_result) -> None:
    """Zero reconciliation breaches — and the residual drift only ever overshoots.

    Two separate mistakes are guarded here, both of which manufacture false breaches
    on a healthy run and thereby hide the real ones:

    1. Comparing a fraction-of-equity target against the *close*-marked position
       reports a breach every time price moves, whether or not anything went wrong.
       The broker snapshots the position at the execution mark instead.
    2. Ignoring the fact that a trade's own cost shrinks the equity its target is a
       fraction *of*. Sizing from pre-cost equity and paying the fee leaves the
       realised fraction larger than the target by ``|target| * cost / equity`` —
       ~1e-3 at full size here, a thousand times the 1e-6 default tolerance. Before
       the runner accounted for it, nine of 480 bars logged a breach, every one of
       them a full-size trade.

    The second assertion pins the *sign*, and only on bars where a fill actually
    happened: cost-induced drift is always outward, because paying money can only
    shrink the denominator. An inward drift on a bar that traded would mean a fill
    that did not fully happen, which is a different and real problem.

    Bars with no fill are excluded rather than waved through. A held position's
    fraction wanders in either direction as the mark moves, and the corrective trade
    is often below the venue's minimum notional and so legitimately unsendable —
    which is precisely what the ``unreachable`` floor in the runner is for. Their
    drift is bounded by the first assertion instead.
    """
    m = paper_result.metrics
    assert m.reconciliation_breaches == 0, (
        f"{m.reconciliation_breaches} breaches, max drift {m.max_abs_drift:.3e}"
    )

    traded = [d for d in paper_result.decisions if abs(float(d["filled_qty"])) > 0.0]
    assert traded, "no bar traded; the sign check below has nothing to check"
    inward = [
        d
        for d in traded
        # Drift and target share a sign when the position overshot outward.
        if abs(float(d["position_drift"])) > 1e-12
        and float(d["position_drift"]) * float(d["executed_target"]) < 0.0
    ]
    assert not inward, (
        f"{len(inward)} of {len(traded)} bars that traded drifted *inward* (the "
        f"realised position fell short of the target rather than overshooting it); "
        f"first at bar {inward[0]['bar']}: target {inward[0]['executed_target']}, "
        f"realised {inward[0]['realized_position']}"
    )


def test_paper_run_cost_identity(paper_result, bundle) -> None:
    from archetype_trader.eval.metrics import cost_attribution

    att = cost_attribution(paper_result.summary)
    assert abs(att["residual"]) < 1e-6 * bundle.cfg.env.initial_cash, att


# --------------------------------------------------------------------------- #
# 5. Parity with the training environment
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def env_reference(bundle, stack):
    """The same decisions, taken through the *training* path instead."""
    from archetype_trader.eval.strategies import make_horizon_env
    from archetype_trader.phase2.train import run_selector
    from archetype_trader.phase3.adapter import make_controller
    from archetype_trader.utils.seeding import seed_everything

    cfg = bundle.cfg
    seed_everything(0)
    override = None
    if stack.phase3 is not None:
        override = make_controller(
            stack.phase3.model,
            grid=cfg.env.positions,
            n_archetypes=stack.n_archetypes,
            budget=cfg.phase3.override_budget,
            horizon=cfg.phase1.horizon,
            device=DEVICE,
        )
    henv = make_horizon_env(
        cfg,
        bundle.data,
        bundle.features,
        bundle.scaler,
        stack.phase1.model,
        stack.scaled,
        bundle.fold.test,
        device=DEVICE,
        override=override,
        shadow_cost_frac=0.0 if stack.dataset is None else stack.dataset.cost_frac,
    )
    stats = run_selector(henv, stack.phase2.policy, device=DEVICE, greedy=True)
    return henv, stats


def test_the_reference_run_actually_trades(paper_result) -> None:
    """Guard the parity tests against passing on an all-cash run.

    Every parity assertion below compares two series. If both are identically zero
    they agree, and the test reports success while proving nothing: a flat run takes
    no fills, pays no fees and has a constant equity curve, so it cannot detect a
    fill priced off the wrong bar or a fee charged at the wrong mark.

    This is not hypothetical. At the fixture's original 4 Phase-1 epochs the decoder
    was behaviourally degenerate and the selector chose the cash archetype for all
    480 bars, and the equity-parity check passed at deviation exactly 0.0 — on two
    flat lines. See the note on ``phase1`` in conftest.
    """
    traded = [d for d in paper_result.decisions if abs(float(d["executed_target"])) > 1e-12]
    assert traded, (
        "the reference paper run never left cash, so every parity assertion in this "
        "module is comparing zeros; raise phase1.epochs in conftest.SMOKE_CONFIG"
    )
    assert paper_result.summary["n_trades"] > 0, paper_result.summary["n_trades"]
    assert paper_result.summary["fees"] > 0.0, "no fees charged despite trading"


def test_model_action_parity(bundle, paper_result, env_reference) -> None:
    """Pre-shield: the deployed agent computes the same decision as training."""
    henv, _ = env_reference
    grid = bundle.cfg.env.positions
    expected = [grid[a] for o in henv.history for a in o.actions]
    actual = [float(d["proposed_position"]) for d in paper_result.decisions]
    assert len(expected) == len(actual), (len(expected), len(actual))
    bad = [i for i in range(len(expected)) if abs(expected[i] - actual[i]) > 1e-12]
    assert not bad, f"{len(bad)} model-action mismatches, first at {bad[:5]}"


def test_executed_action_parity(paper_result, env_reference) -> None:
    """Post-shield and post-latency: the runner's gates are ordered like the env's."""
    from archetype_trader.live import action_sequence

    henv, _ = env_reference
    expected = [float(p) for o in henv.history for p in o.positions]
    actual = action_sequence(paper_result)
    assert len(expected) == len(actual), (len(expected), len(actual))
    bad = [i for i in range(len(expected)) if abs(expected[i] - actual[i]) > 1e-12]
    assert not bad, f"{len(bad)} executed-action mismatches, first at {bad[:5]}"


def test_archetype_sequence_parity(paper_result, env_reference) -> None:
    henv, _ = env_reference
    expected = [o.archetype for o in henv.history]
    actual = [d["archetype"] for d in paper_result.decisions if d["new_horizon"]]
    assert expected == actual


def test_broker_reproduces_the_training_env(paper_result, env_reference) -> None:
    """Exact equity parity. Deviation must be 0.0, not merely small.

    A tolerance here would conceal a fee charged at a different mark or a fill
    priced off the close rather than the open — the bugs that make a backtest
    optimistic in a way no summary statistic reveals.
    """
    _, stats = env_reference
    env_eq = np.asarray(stats["equity_curve"], dtype=np.float64)
    paper_eq = np.asarray(paper_result.summary["equity_curve"], dtype=np.float64)
    assert env_eq.shape == paper_eq.shape, (env_eq.shape, paper_eq.shape)
    deviation = float(np.max(np.abs(env_eq - paper_eq)))
    assert deviation == 0.0, f"max |equity deviation| = {deviation:.3e}"


def test_paper_run_is_deterministic(bundle, stack, paper_result) -> None:
    from archetype_trader.live import action_sequence
    from archetype_trader.utils.seeding import seed_everything

    seed_everything(0)
    again = _make_runner(bundle, stack, run_id="paper-test").run()
    assert action_sequence(again) == action_sequence(paper_result)
    assert again.final_equity == paper_result.final_equity


# --------------------------------------------------------------------------- #
# 6. Operator limits and the kill switch
# --------------------------------------------------------------------------- #
def test_each_operator_cap_clamps_and_names_itself(bundle, stack) -> None:
    """Unit-test the limits gate on a synthetic account, one cap at a time.

    Deliberately *not* driven through a trained model. Whether the agent happens to
    want a position large enough to breach a cap is a property of the fixture's
    RNG, and an end-to-end test that asserted ``limit_blocks > 0`` would pass or
    fail depending on which stack got trained — which is how this test first failed
    on a stack whose first sixty bars were entirely flat.

    Each cap is checked in isolation, and each must name itself in the reason
    tuple: an unattributed clamp is indistinguishable from a model decision when
    reading the journal after the fact.
    """
    from dataclasses import replace

    from archetype_trader.live import LiveLimits

    runner = _make_runner(bundle, stack, run_id="limits-unit")
    account = replace(runner.broker.account(), equity=100_000.0, position=0.0)

    # Leverage cap: 1.0 requested, 0.4 allowed.
    runner.limits = LiveLimits(max_leverage=0.4)
    out, why = runner._apply_limits(1.0, account=account)
    assert out == pytest.approx(0.4), out
    assert any("max_leverage" in r for r in why), why

    # Position-notional cap: 20% of a 100k account.
    runner.limits = LiveLimits(max_position_notional=20_000.0)
    out, why = runner._apply_limits(1.0, account=account)
    assert out == pytest.approx(0.2), out
    assert any("max_position_notional" in r for r in why), why

    # Order-notional cap throttles the *change*, so it depends on where we start.
    runner.limits = LiveLimits(max_order_notional=10_000.0)
    out, why = runner._apply_limits(1.0, account=replace(account, position=0.5))
    assert out == pytest.approx(0.6), out
    assert any("max_order_notional" in r for r in why), why

    # Symbol allow-list flattens outright rather than clamping.
    runner.limits = LiveLimits(allowed_symbols=("SOMETHING-ELSE",))
    out, why = runner._apply_limits(1.0, account=account)
    assert out == 0.0, out
    assert any("symbol_not_allowed" in r for r in why), why

    # And a request inside every cap is passed through untouched, with no reason.
    runner.limits = LiveLimits(
        max_leverage=2.0, max_position_notional=200_000.0, max_order_notional=200_000.0
    )
    out, why = runner._apply_limits(0.5, account=account)
    assert out == 0.5 and why == (), (out, why)


def test_operator_caps_are_never_exceeded_end_to_end(bundle, stack, paper_result) -> None:
    """A cap set below what the model actually asked for must bind, and hold.

    The cap is *derived* from ``paper_result`` — the same run without limits — at
    half the largest position that run took. A hard-coded cap would be vacuous on
    any stack that happened to trade smaller than it, which is exactly how this test
    first failed: a fixture whose first sixty bars were entirely flat.
    """
    from archetype_trader.live import LiveLimits

    unlimited_peak = max(abs(d["executed_target"]) for d in paper_result.decisions)
    if unlimited_peak == 0.0:
        pytest.skip("the unlimited run never left cash, so no cap can bind")

    cap = unlimited_peak / 2.0
    res = _make_runner(bundle, stack, run_id="limited", limits=LiveLimits(max_leverage=cap)).run()
    peak = max(abs(d["executed_target"]) for d in res.decisions)
    assert peak <= cap + 1e-12, f"executed {peak} above cap {cap}"
    assert res.metrics.limit_blocks > 0, (
        f"the unlimited run reached {unlimited_peak} but the run capped at {cap} recorded "
        "no limit_blocks; the gate clamped without journalling"
    )


def test_kill_switch_forces_and_holds_flat(bundle, stack) -> None:
    cfg = bundle.cfg
    killed = cfg.shield.model_copy(update={"kill_switch": True})
    res = _make_runner(bundle, stack, run_id="killed", shield_cfg=killed).run(max_bars=40)
    peak = max(abs(d["executed_target"]) for d in res.decisions)
    assert peak < 1e-12, peak
    # Never traded, so equity is untouched to the cent.
    assert res.final_equity == cfg.env.initial_cash


def test_flatten_closes_the_position_immediately(bundle) -> None:
    from archetype_trader.live.broker import Order

    b = _broker(bundle, offset=60)
    b.submit(Order(client_order_id="x1", symbol="S", target_position=1.0, decided_at=b.index))
    b.advance()
    assert abs(b.ledger.position_frac) > 0.5, b.ledger.position_frac

    fill = b.flatten(reason="kill_switch")
    assert fill is not None
    assert abs(b.ledger.qty) < 1e-12, b.ledger.qty
