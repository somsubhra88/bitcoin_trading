"""Portfolio accounting identities.

A backtest that gets the arithmetic wrong is worse than one that gets the strategy
wrong: the strategy failure is visible, the arithmetic failure inflates every
number downstream and is invisible. These tests pin the identities that must hold
after every single bar, not merely at the end of a run — an error that cancels out
by the last bar is still an error, and will not cancel on different data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from tests.archetype.conftest import DEVICE


def _env(bundle, *, shield_enabled: bool = True):
    """Build a TradingEnv over the test segment.

    Reuses ``strategies._make_env`` rather than reconstructing the start index.
    That arithmetic (``max(segment.start, valid_from, lookback - 1, context_bars - 1)``)
    is precisely where an off-by-one would hide, and a test that copies it would
    agree with its own copy rather than with the code under test.
    """
    from archetype_trader.eval.strategies import _make_env

    return _make_env(
        bundle.cfg,
        bundle.data,
        bundle.features,
        bundle.scaler,
        bundle.fold.test,
        shield_enabled=shield_enabled,
    )


def test_ledger_identity_holds_after_every_bar(bundle) -> None:
    """``equity == cash + qty * mark`` at every step, not just the last.

    Checked to a relative tolerance because equity is O(1e5) and float64 carries
    ~1e-11 relative error; an absolute 1e-9 bound would really be a test of how
    large ``initial_cash`` happens to be.
    """
    env = _env(bundle)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    n_actions = len(bundle.cfg.env.positions)
    n = 0
    while True:
        _obs, _r, term, trunc, _info = env.step(int(rng.integers(n_actions)))
        led = env.ledger
        identity = led.cash + led.qty * led.mark
        assert identity == pytest.approx(led.equity, rel=1e-10, abs=1e-6), (
            f"bar {n}: cash {led.cash} + qty {led.qty} * mark {led.mark} = {identity} "
            f"!= equity {led.equity}"
        )
        assert np.isfinite(led.equity), (n, led.equity)
        n += 1
        if term or trunc:
            break
    assert n > 10, f"episode ended after {n} bars; the test proves little"


def test_gross_minus_costs_equals_net(bundle) -> None:
    """The cost attribution residual must vanish, through the real function.

    Deliberately calls :func:`eval.metrics.cost_attribution` rather than
    re-deriving the identity here. A hand-rolled copy in a test is a second
    implementation that can agree with neither the code nor reality — this exact
    mistake produced a spurious 56.6-unit residual during development, because the
    test treated funding as a credit where the code treats it as a cost.
    """
    from archetype_trader.eval.metrics import cost_attribution

    env = _env(bundle)
    env.reset(seed=0)
    # Alternate flat and maximum long so fees, spread and slippage are all
    # non-zero: an all-cash run satisfies the identity trivially.
    n_actions = len(bundle.cfg.env.positions)
    flat = bundle.cfg.env.positions.index(0.0)
    step = 0
    while True:
        action = n_actions - 1 if (step // 20) % 2 == 0 else flat
        _o, _r, term, trunc, _i = env.step(action)
        step += 1
        if term or trunc:
            break

    summary = env.episode_summary()
    att = cost_attribution(summary)
    assert att["fees"] != 0.0, "no fees were charged; the identity is trivially satisfied"
    assert abs(att["residual"]) < 1e-6 * bundle.cfg.env.initial_cash, att


def test_recorded_positions_are_the_realised_fraction_not_the_target(bundle) -> None:
    """Pin the difference between the *target* sent and the *fraction* realised.

    ``info["executed_position"]`` is the target that cleared the latency queue: a
    grid value such as 0.5. ``episode_summary()["positions"]`` is
    ``ledger.position_frac`` after the fill and the re-mark to the close: 0.499902.
    They are different quantities and the gap is real — execution cost, plus the
    fact that a target is a *fraction of equity* and equity moves between the fill
    mark and the close mark.

    Conflating the two is not hypothetical. It is what made paper-trading
    reconciliation report hundreds of spurious breaches during development, until
    ``PaperBroker`` began snapshotting the position at the execution mark. So this
    test asserts both halves: the series are aligned bar for bar, and they differ
    only by an amount attributable to one bar of cost and drift.

    Note also that ``turnover`` in the metrics table is *notional* turnover over
    initial equity, not Σ|Δposition|; only the notional figure is charged fees.
    """
    env = _env(bundle)
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    n_actions = len(bundle.cfg.env.positions)
    executed = []
    while True:
        _o, _r, term, trunc, info = env.step(int(rng.integers(n_actions)))
        executed.append(float(info["executed_position"]))
        if term or trunc:
            break

    summary = env.episode_summary()
    recorded = np.asarray(summary["positions"], dtype=np.float64)
    # The series is seeded at ``reset`` with the flat starting position, so it runs
    # one longer than the number of steps. ``equity_curve`` is seeded the same way;
    # this is the alignment the paper broker has to reproduce for the equity-parity
    # check in test_live.py to mean anything.
    assert recorded.shape == (len(executed) + 1,), (recorded.shape, len(executed))
    assert recorded[0] == 0.0, recorded[0]

    target = np.asarray(executed, dtype=np.float64)
    realised = recorded[1:]

    # Aligned: each realised fraction must sit nearer its own bar's target than to
    # any other grid point. This is what catches an off-by-one, which a pure
    # tolerance check would happily absorb.
    grid = np.asarray(bundle.cfg.env.positions, dtype=np.float64)
    nearest = grid[np.argmin(np.abs(realised[:, None] - grid[None, :]), axis=1)]
    bad = np.flatnonzero(nearest != target)
    assert bad.size == 0, (
        f"{bad.size} bars where the realised fraction snaps to a different grid "
        f"point than the target; first at step {int(bad[0])}: realised "
        f"{realised[bad[0]]:.6f} -> {nearest[bad[0]]} but target {target[bad[0]]}"
    )

    # And close: the drift is one bar of cost and mark movement, not a regime.
    drift = np.abs(realised - target)
    assert drift.max() < 0.05, f"max |realised - target| = {drift.max():.4f}"

    # Turnover was charged if and only if the position actually moved.
    moved = float(np.abs(np.diff(recorded)).sum()) > 0.0
    assert (float(summary["turnover_notional"]) > 0.0) == moved


def test_a_never_trading_policy_pays_nothing(bundle) -> None:
    """Holding cash must cost exactly zero in fees, slippage and turnover.

    The floor every strategy is measured against. If cash leaks cost, every
    reported excess return is inflated by that leak.
    """
    env = _env(bundle)
    env.reset(seed=0)
    flat = bundle.cfg.env.positions.index(0.0)
    while True:
        _o, _r, term, trunc, _i = env.step(flat)
        if term or trunc:
            break
    s = env.episode_summary()
    assert s["fees"] == 0.0, s["fees"]
    assert s["execution_cost"] == 0.0, s["execution_cost"]
    assert s["turnover_notional"] == 0.0, s["turnover_notional"]
    assert s["n_trades"] == 0, s["n_trades"]
    assert s["funding"] == 0.0, "funding was charged on a zero position"
    assert s["final_equity"] == pytest.approx(bundle.cfg.env.initial_cash, rel=0, abs=1e-9)


def test_equity_never_goes_negative_without_a_liquidation(bundle) -> None:
    """A leveraged account may be liquidated; it may not silently go negative.

    Unbounded negative equity means the maintenance-margin check never fired,
    which in a backtest presents as a spectacular recovery from an impossible
    state rather than as an error.
    """
    assert bundle.cfg.env.mode == "perp", "this test exercises the perp margin path"
    env = _env(bundle)
    env.reset(seed=0)
    max_long = len(bundle.cfg.env.positions) - 1
    liquidated = False
    while True:
        _o, _r, term, trunc, info = env.step(max_long)
        liquidated = liquidated or bool(info.get("liquidated"))
        assert env.ledger.equity >= 0.0 or liquidated, (
            f"equity {env.ledger.equity} < 0 with no liquidation flag raised"
        )
        if term or trunc:
            break


def test_cost_accumulators_are_monotone(bundle) -> None:
    """Fees, execution cost and turnover only ever increase.

    They are cumulative charges, so a decrease means something is being refunded.
    Funding is excluded: on a perp it is genuinely signed and may be received.
    """
    env = _env(bundle)
    env.reset(seed=0)
    rng = np.random.default_rng(2)
    n_actions = len(bundle.cfg.env.positions)
    prev = (0.0, 0.0, 0.0)
    while True:
        _o, _r, term, trunc, _i = env.step(int(rng.integers(n_actions)))
        led = env.ledger
        now = (led.cum_fees, led.cum_execution_cost, led.cum_turnover_notional)
        for name, a, b in zip(
            ("cum_fees", "cum_execution_cost", "cum_turnover_notional"), prev, now, strict=True
        ):
            assert b >= a - 1e-12, f"{name} decreased from {a} to {b}"
        prev = now
        if term or trunc:
            break


# --------------------------------------------------------------------------- #
# Phase-3 counterfactual accounting
# --------------------------------------------------------------------------- #
@dataclass
class _Decision:
    action: int


class _RetainAll:
    """A ``StepOverride`` that never overrides. Structural, not a subclass.

    Implemented against the Protocol rather than by patching the trained adapter,
    because the claim under test is about the *environment's* counterfactual
    bookkeeping, not about what the adapter chose.
    """

    def reset(self, archetype: int) -> None:
        self.calls = 0

    def decide(self, market_row: np.ndarray, *, base_action: int, **_: Any) -> _Decision:
        self.calls = getattr(self, "calls", 0) + 1
        return _Decision(action=int(base_action))


def test_no_override_means_exactly_zero_improvement(bundle, stack) -> None:
    """Phase-3 improvement must be *exactly* 0.0 when nothing is overridden.

    This is what the difference-based counterfactual accounting buys. The base
    plan's return is anchored to the realised net return, and only the position
    and turnover *difference* is shadowed. Were improvement computed by
    re-simulating the base plan independently, retaining everywhere would yield a
    small non-zero number from re-marking, and every Phase-3 result would sit on
    top of that noise floor. Exact equality is the assertion that the anchor holds.
    """
    from archetype_trader.eval.strategies import make_horizon_env
    from archetype_trader.phase2.train import run_selector

    if stack.phase2 is None:
        pytest.skip("no phase-2 selector trained")

    retain = _RetainAll()
    henv = make_horizon_env(
        bundle.cfg,
        bundle.data,
        bundle.features,
        bundle.scaler,
        stack.phase1.model,
        stack.scaled,
        bundle.fold.test,
        device=DEVICE,
        override=retain,
        shadow_cost_frac=0.0 if stack.dataset is None else stack.dataset.cost_frac,
    )
    run_selector(henv, stack.phase2.policy, device=DEVICE, greedy=True)

    assert henv.history, "no horizons ran"
    assert retain.calls > 0, "the override was never consulted; the test is vacuous"
    worst = max(abs(o.improvement) for o in henv.history)
    assert worst == 0.0, (
        f"a retain-everywhere override produced non-zero improvement (max |imp| = "
        f"{worst:.3e}); the base-plan return is not anchored to the realised return"
    )
    assert all(o.n_overrides == 0 for o in henv.history)
    assert all(o.actions == o.base_actions for o in henv.history)


def test_override_budget_is_never_exceeded(bundle, stack) -> None:
    """The adapter may open at most ``override_budget`` contiguous override runs.

    Counting *runs* rather than bars is the whole point of the budget: holding an
    override for ten consecutive bars is one decision to deviate, not ten.
    """
    from archetype_trader.eval.strategies import make_horizon_env
    from archetype_trader.phase2.train import run_selector
    from archetype_trader.phase3.adapter import make_controller

    if stack.phase3 is None or stack.phase2 is None:
        pytest.skip("no phase-3 adapter trained")

    budget = bundle.cfg.phase3.override_budget
    ctrl = make_controller(
        stack.phase3.model,
        grid=bundle.cfg.env.positions,
        n_archetypes=stack.n_archetypes,
        budget=budget,
        horizon=bundle.cfg.phase1.horizon,
        device=DEVICE,
    )
    henv = make_horizon_env(
        bundle.cfg,
        bundle.data,
        bundle.features,
        bundle.scaler,
        stack.phase1.model,
        stack.scaled,
        bundle.fold.test,
        device=DEVICE,
        override=ctrl,
        shadow_cost_frac=0.0 if stack.dataset is None else stack.dataset.cost_frac,
    )
    run_selector(henv, stack.phase2.policy, device=DEVICE, greedy=True)

    assert henv.history
    for i, o in enumerate(henv.history):
        assert o.n_overrides <= budget, f"horizon {i}: {o.n_overrides} runs > budget {budget}"
        # Recount from the action traces, so the assertion does not trust the
        # counter it is checking.
        runs = 0
        inside = False
        for base, taken in zip(o.base_actions, o.actions, strict=True):
            if taken != base:
                runs += int(not inside)
                inside = True
            else:
                inside = False
        assert runs == o.n_overrides, (i, runs, o.n_overrides)
        assert runs <= budget, (i, runs, budget)


def test_improvement_is_zero_exactly_when_nothing_overrode(bundle, stack) -> None:
    """The converse direction: a horizon with overrides may gain or lose, but a
    horizon without them must be exactly zero, per horizon rather than on average.

    An average of zero can hide equal and opposite bookkeeping errors.
    """
    from archetype_trader.eval.strategies import make_horizon_env
    from archetype_trader.phase2.train import run_selector
    from archetype_trader.phase3.adapter import make_controller

    if stack.phase3 is None or stack.phase2 is None:
        pytest.skip("no phase-3 adapter trained")

    ctrl = make_controller(
        stack.phase3.model,
        grid=bundle.cfg.env.positions,
        n_archetypes=stack.n_archetypes,
        budget=bundle.cfg.phase3.override_budget,
        horizon=bundle.cfg.phase1.horizon,
        device=DEVICE,
    )
    henv = make_horizon_env(
        bundle.cfg,
        bundle.data,
        bundle.features,
        bundle.scaler,
        stack.phase1.model,
        stack.scaled,
        bundle.fold.test,
        device=DEVICE,
        override=ctrl,
        shadow_cost_frac=0.0 if stack.dataset is None else stack.dataset.cost_frac,
    )
    run_selector(henv, stack.phase2.policy, device=DEVICE, greedy=True)

    for i, o in enumerate(henv.history):
        if o.n_overrides == 0:
            assert o.improvement == 0.0, (i, o.improvement)
            assert o.net_log_return == o.base_net_log_return, (i,)
