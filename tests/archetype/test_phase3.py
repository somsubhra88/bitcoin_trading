"""Phase 3: the hindsight target, the override budget, and controller parity.

Phase 3 is the part of the stack most easily fooled by its own bookkeeping. It is
trained against a *hindsight* target produced by a DP that sees the whole horizon,
and it is judged by an *improvement* over a counterfactual base plan that never
actually ran. Get either wrong and the adapter reports gains it did not make.

Four properties are pinned here.

**The DP is right.** :func:`best_override` is checked against an exhaustive search
over every feasible action sequence on small horizons. This is the only assertion
in the file that does not depend on a trained model, and it is the one that would
catch a wrong recurrence rather than merely a wrong number.

**The training target is achievable.** Non-negative headroom and a target within
budget. A label that violated the budget would teach the adapter to want something
it is structurally forbidden to do at inference.

**The two inference paths agree.** :func:`rollout_batch` (vectorised; used in
training and in ``evaluate_adapter``, and therefore the source of every reported
Phase-3 number) and :class:`AdapterController` (one step at a time; used by
``HorizonEnv`` and the live agent, and therefore the deployed path) must choose
identical actions. They share weights, not code.

**The budget binds at inference**, structurally — via logit masking — and not
merely on average.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.archetype.conftest import DEVICE

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def dataset(stack):
    """The Phase-3 horizon dataset built during ``stack`` training.

    Reused rather than rebuilt so the ``base_actions`` here are the plans of the
    very decoder the ``adapter`` fixture was fitted against. Rebuilding would
    give a dataset the adapter had never seen and quietly weaken the parity and
    budget tests into statements about an unrelated pair of objects.
    """
    if stack.dataset is None:
        pytest.skip("stack was trained without phase 3")
    return stack.dataset


@pytest.fixture(scope="module")
def phase3(stack):
    if stack.phase3 is None:
        pytest.skip("stack was trained without phase 3")
    return stack.phase3


@pytest.fixture(scope="module")
def adapter(phase3):
    return phase3.model


# --------------------------------------------------------------------------- #
# The DP itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("grid", "horizon", "budget"),
    [
        ((-1.0, 0.0, 1.0), 9, 1),
        ((-1.0, 0.0, 1.0), 9, 2),
        ((-1.0, -0.5, 0.0, 0.5, 1.0), 6, 1),
        ((-1.0, -0.5, 0.0, 0.5, 1.0), 6, 2),
    ],
)
def test_the_dp_matches_exhaustive_search(grid, horizon, budget) -> None:
    """``best_override`` must equal the maximum over *every* feasible sequence.

    The value is compared, not the action sequence: ties are common on a coarse
    grid and the DP is free to break them differently from ``itertools.product``.
    Comparing values is the property that matters — an argmax that finds an equally
    good plan is correct, a value below the true optimum is a broken recurrence.

    Random base plans and returns, several grids and budgets, and a *large* cost
    (25 bps) so the turnover charge actually binds: with negligible costs the
    optimum degenerates to "sign of the return" and a DP that mishandles the
    override-run bookkeeping would still get the right answer.
    """
    from archetype_trader.phase3 import best_override
    from archetype_trader.phase3.hindsight import brute_force_best_override

    rng = np.random.default_rng(11)
    n = 12
    base = rng.integers(0, len(grid), size=(n, horizon)).astype(np.int64)
    returns = rng.normal(0.0, 0.01, size=(n, horizon))
    kw = {"grid": grid, "cost_frac": 25e-4, "budget": budget, "init_position": 0.0}

    dp = best_override(base, returns, **kw)
    _, brute_value = brute_force_best_override(base, returns, **kw)

    worst = float(np.max(np.abs(dp.value - brute_value)))
    assert worst < 1e-12, (
        f"DP disagrees with exhaustive search by up to {worst:.3e}; "
        f"first offender at row {int(np.argmax(np.abs(dp.value - brute_value)))}"
    )


def test_prohibitive_cost_makes_the_dp_retain_a_still_plan() -> None:
    """A base plan that already holds still cannot be improved when trading is dear.

    The two halves of this test are the two ways the cost term can be wrong, and
    they fail in *opposite* directions:

    * A base plan that never moves pays no turnover, so with an edge far below the
      fee there is nothing an override can buy. If the cost were dropped from the
      branch that starts an override, the DP would find a free lunch here and
      report headroom on a horizon that has none — headroom the adapter is then
      trained to chase and can never reach.
    * A base plan that churns every bar pays that fee ten times over, so the best
      bounded adaptation is not to trade *better* but to **stop trading**. If the
      cost were dropped from the branch that continues an override, the DP would
      undervalue exactly this move. The optimum is the flat action held for the
      whole horizon, bought with a single override run.

    The second half is the one that corrects the intuition worth stating plainly:
    at a prohibitive cost the hindsight ceiling is a *lower*-turnover plan, not a
    more aggressive one.
    """
    from archetype_trader.phase3 import best_override

    grid = (-1.0, 0.0, 1.0)
    flat = 1  # index of 0.0 in the grid
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 1e-5, size=(8, 10))  # edge orders of magnitude below the fee
    kw = {"grid": grid, "cost_frac": 0.05, "budget": 2, "init_position": 0.0}

    still = np.full((8, 10), flat, dtype=np.int64)
    res = best_override(still, returns, **kw)
    assert np.array_equal(res.actions, still), "an override paid for itself against a still plan"
    assert float(np.abs(res.headroom).max()) < 1e-12
    assert res.n_overrides.max() == 0

    churn = np.tile(np.array([0, 2] * 5, dtype=np.int64), (8, 1))
    res = best_override(churn, returns, **kw)
    assert np.array_equal(res.actions, still), (
        "the best bounded adaptation of a plan churning at a 5% cost is to go flat "
        f"and stay flat; got {res.actions[0].tolist()}"
    )
    assert (res.n_overrides == 1).all(), res.n_overrides
    assert float(res.headroom.min()) > 0.0, "flattening a churning plan saved nothing"


def test_count_overrides_counts_runs_not_bars() -> None:
    """The budget means *runs*, not bars. Pinned on hand-built cases.

    Holding an override for ten consecutive bars is one decision to deviate. If
    this counted bars, ``override_budget: 1`` would forbid anything but a
    single-bar blip and the Phase-3 ablation would be measuring the wrong knob.
    """
    from archetype_trader.phase3 import count_overrides

    base = np.zeros((5, 5), dtype=np.int64)
    taken = np.array(
        [
            [0, 0, 0, 0, 0],  # identical
            [0, 1, 1, 1, 0],  # one 3-bar run
            [1, 0, 1, 0, 0],  # two runs, separated by a retained bar
            [1, 0, 1, 0, 1],  # alternating
            [1, 1, 1, 1, 1],  # one run spanning the whole horizon
        ],
        dtype=np.int64,
    )
    assert count_overrides(taken, base).tolist() == [0, 1, 2, 3, 1]


# --------------------------------------------------------------------------- #
# The training target
# --------------------------------------------------------------------------- #
def test_hindsight_headroom_is_never_negative(dataset) -> None:
    """The best bounded override cannot be worse than retaining everywhere.

    Not redundant with the assertion inside ``best_override``. That one compares
    the optimum against the base value the DP computed itself; this compares it
    against ``dataset.base_value``, which ``build_horizon_dataset`` recomputes
    independently via ``plan_value``. A sign slip or an off-by-one in the
    turnover term of either would show up here and nowhere else — and it would
    show up as free headroom the adapter is then trained to chase.
    """
    worst = float(dataset.headroom.min())
    assert worst >= -1e-9, (
        f"minimum hindsight headroom is {worst:.3e}: the DP's optimum scores below the "
        "independently computed base value, so the two disagree on costs"
    )


def test_the_hindsight_target_respects_the_budget(dataset) -> None:
    """The label the adapter imitates must itself be reachable at inference."""
    from archetype_trader.phase3 import count_overrides

    runs = count_overrides(dataset.best_actions, dataset.base_actions)
    over = np.flatnonzero(runs > dataset.budget)
    assert over.size == 0, (
        f"{over.size} hindsight targets exceed the budget of {dataset.budget}; worst is "
        f"{int(runs.max())} runs, first at episode {int(over[0])}"
    )


def test_the_retain_label_is_used_where_and_only_where_the_plan_is_kept(dataset) -> None:
    """``target_classes`` must encode retention as the extra class, everywhere.

    If retention were instead labelled with the base action's own index, the same
    intent would carry a different label in different horizons, the imitation
    cross-entropy would be fitting an inconsistent target, and the budget could
    not be enforced by masking one logit.
    """
    retain = len(dataset.grid)
    kept = dataset.best_actions == dataset.base_actions
    assert np.array_equal(dataset.target_classes == retain, kept), (
        "target_classes and (best == base) disagree about which steps retain"
    )
    changed = ~kept
    assert np.array_equal(dataset.target_classes[changed], dataset.best_actions[changed]), (
        "an overriding step's label is not the action it overrides to"
    )


def test_the_dataset_never_reads_past_the_training_segment(bundle, dataset) -> None:
    """Every episode, horizon included, ends before validation begins.

    ``best_actions`` is hindsight and ``returns`` spans the whole horizon, so a
    single episode straddling ``val.start`` would put knowledge of the validation
    segment into the adapter's supervision — and Phase 3's headline number is a
    validation improvement.
    """
    fold = bundle.fold
    assert int(dataset.starts.min()) >= fold.train.start, (
        int(dataset.starts.min()),
        fold.train.start,
    )
    # The planner's last return is taken from ``open[start + h + 1]``.
    last_read = int(dataset.starts.max()) + dataset.horizon + 1
    assert last_read < fold.train.stop <= fold.val.start, (
        last_read,
        fold.train.stop,
        fold.val.start,
    )


# --------------------------------------------------------------------------- #
# Controller vs batched rollout
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def eager_adapter(adapter):
    """A deliberately override-happy adapter: same shape, randomised policy head.

    Necessary, not decorative. **The trained adapter retains on every step of
    every horizon on this fixture** — measured ``override_rate`` 0.0 at all six
    epochs, ``regret_ratio`` 1.0 against a mean hindsight headroom of +0.062 log
    return per horizon. That is not a bug: overriding genuinely loses money on
    this low-signal synthetic data, and the selection objective is greedy
    validation improvement, so the checkpoint that comes back is an early all-retain
    one. Raising ``phase3.epochs`` cannot change it.

    But it makes every parity and budget assertion below vacuous. If the adapter
    always retains, ``rollout_batch`` and :class:`AdapterController` both just echo
    ``base_actions``, and they would agree even if the recurrent state, the feature
    layout, the masking and the scalar bookkeeping were all implemented
    differently in the two paths. Likewise a budget cannot be observed to bind
    against a policy that never spends it.

    Randomising the policy head — and flipping the retain bias negative — buys a
    policy that overrides constantly. Parity and the budget mask are properties of
    the *code*, so they must hold for arbitrary weights;
    ``test_the_eager_adapter_actually_overrides`` fails if this fixture ever stops
    producing overrides.
    """
    import copy

    eager = copy.deepcopy(adapter)
    gen = torch.Generator().manual_seed(20260906)
    with torch.no_grad():
        eager.policy.weight.copy_(torch.randn(eager.policy.weight.shape, generator=gen) * 1.5)
        eager.policy.bias.copy_(torch.randn(eager.policy.bias.shape, generator=gen))
        eager.policy.bias[eager.retain_index] = -6.0
    eager.eval()
    return eager


@pytest.fixture(scope="module", params=["trained", "eager"])
def any_adapter(request, adapter, eager_adapter):
    """Every weight-independent property is checked on both policies.

    ``trained`` is what ships; ``eager`` is what makes the assertion mean
    something. See :func:`eager_adapter`.
    """
    return {"trained": adapter, "eager": eager_adapter}[request.param]


def test_the_eager_adapter_actually_overrides(cfg, dataset, eager_adapter) -> None:
    """Guard against the parity and budget tests going quietly vacuous.

    If this fails, the ``eager`` half of every parametrised test below has silently
    degenerated into a second copy of the ``trained`` half, which retains
    everywhere — and the tests will still pass while proving nothing.
    """
    from archetype_trader.phase3 import evaluate_adapter

    rows = np.arange(len(dataset))
    ev = evaluate_adapter(cfg, eager_adapter, dataset, rows, device=DEVICE)
    assert ev["override_rate"] > 0.5, (
        f"the eager adapter overrode on only {ev['override_rate']:.1%} of horizons; "
        "re-tune the randomised policy head in the eager_adapter fixture"
    )
    assert ev["max_overrides"] == dataset.budget, (
        f"the eager adapter used at most {ev['max_overrides']} of {dataset.budget} "
        "overrides, so the budget mask is never actually reached"
    )


def test_the_controller_matches_the_batched_rollout(dataset, any_adapter) -> None:
    """Step-at-a-time and vectorised inference must choose the same action, always.

    ``rollout_batch`` produces every Phase-3 number that gets reported;
    :class:`AdapterController` is what actually trades. A divergence means the
    metrics table describes a policy that is not the one deployed. Run against both
    the trained policy and an override-happy one, because the trained policy retains
    everywhere on this fixture and would make the comparison trivial.

    The cumulative-return scalars fed to the controller are recomputed here with
    **torch float32 ops in the rollout's own order**, not with float64 NumPy. That
    is deliberate: the rollout accumulates ``real_cum`` in float32, and
    accumulating the same sum in float64 and casting at the end differs in the last
    bits. Those bits reach an ``argmax`` over logits, so a float64 reimplementation
    makes this test flake on ties rather than testing anything. Matching the dtype
    is what turns "usually agrees" into "agrees exactly".
    """
    from archetype_trader.phase3 import make_controller, rollout_batch

    adapter = any_adapter
    rows = np.arange(min(24, len(dataset)))
    pos = torch.tensor(dataset.grid, dtype=torch.float32)
    rets = torch.from_numpy(np.ascontiguousarray(dataset.returns[rows])).float()
    init_action = int(np.argmin(np.abs(np.asarray(dataset.grid))))
    cost_frac = dataset.cost_frac

    batched = rollout_batch(
        adapter,
        torch.from_numpy(dataset.obs_batch(rows)).float(),
        torch.from_numpy(np.ascontiguousarray(dataset.base_actions[rows])),
        rets,
        torch.from_numpy(np.ascontiguousarray(dataset.archetypes[rows])),
        positions=pos,
        cost_frac=cost_frac,
        budget=dataset.budget,
        n_archetypes=dataset.n_archetypes,
        init_action=init_action,
        greedy=True,
    )
    ctrl = make_controller(
        adapter,
        grid=dataset.grid,
        n_archetypes=dataset.n_archetypes,
        budget=dataset.budget,
        horizon=dataset.horizon,
        device=DEVICE,
    )

    mismatches: list[tuple[int, int, int, int]] = []
    for i, row in enumerate(rows):
        ctrl.reset(int(dataset.archetypes[row]))
        prev = base_prev = init_action
        real_cum = base_cum = torch.zeros((), dtype=torch.float32)
        for t in range(dataset.horizon):
            base_action = int(dataset.base_actions[row, t])
            decision = ctrl.decide(
                dataset.features[int(dataset.starts[row]) + t],
                base_action=base_action,
                position=float(pos[prev]),
                realized_cum=float(real_cum),
                base_cum=float(base_cum),
            )
            want = int(batched.actions[i, t])
            if decision.action != want:
                mismatches.append((int(row), t, decision.action, want))
                break  # the recurrent state has diverged; later steps are noise
            a = decision.action
            real_cum = (
                real_cum
                + pos[a] * rets[i, t]
                + torch.log1p(-cost_frac * (pos[a] - pos[prev]).abs())
            )
            base_cum = (
                base_cum
                + pos[base_action] * rets[i, t]
                + torch.log1p(-cost_frac * (pos[base_action] - pos[base_prev]).abs())
            )
            prev, base_prev = a, base_action

    assert not mismatches, (
        f"{len(mismatches)} of {len(rows)} episodes diverged; first "
        f"(episode, step, controller, batched) = {mismatches[0]}"
    )


def test_retain_is_an_extra_class_not_a_reused_one(dataset, adapter) -> None:
    """The head emits ``A + 1`` logits and retention is the last of them."""
    assert adapter.n_actions == len(dataset.grid), (adapter.n_actions, len(dataset.grid))
    assert adapter.retain_index == len(dataset.grid)
    assert adapter.policy.out_features == len(dataset.grid) + 1


def test_an_untrained_adapter_retains_everywhere(cfg, dataset) -> None:
    """Initialisation must be a no-op policy, not a random one.

    An untrained adapter that overrides at random is strictly worse than the
    archetype it wraps, and it would spend the whole budget in the first few bars
    of every horizon before any signal exists. The zeroed policy weights and the
    positive retain bias are load-bearing, not cosmetic.
    """
    from archetype_trader.phase3 import OverrideAdapter, evaluate_adapter

    fresh = OverrideAdapter(
        int(dataset.features.shape[1]), len(dataset.grid), dataset.n_archetypes, hidden_dim=32
    ).to(DEVICE)
    rows = np.arange(min(32, len(dataset)))
    ev = evaluate_adapter(cfg, fresh, dataset, rows, device=DEVICE)  # cfg unused when greedy
    assert ev["override_rate"] == 0.0, ev["override_rate"]
    assert ev["improvement"] == 0.0, (
        f"an all-retain adapter improved by {ev['improvement']:.3e}; the counterfactual "
        "base return is not being priced on identical terms to the realised one"
    )


def test_budget_is_honoured_at_greedy_inference(cfg, dataset, any_adapter) -> None:
    from archetype_trader.phase3 import evaluate_adapter

    rows = np.arange(min(64, len(dataset)))
    ev = evaluate_adapter(cfg, any_adapter, dataset, rows, device=DEVICE)
    assert ev["max_overrides"] <= dataset.budget, (ev["max_overrides"], dataset.budget)
    assert 0.0 <= ev["override_rate"] <= 1.0, ev["override_rate"]
    assert 0.0 <= ev["target_accuracy"] <= 1.0, ev["target_accuracy"]


def test_a_zero_budget_controller_can_never_override(dataset, any_adapter) -> None:
    """With no budget, masking must leave *any* adapter no way to deviate.

    This is what makes ``max_overrides <= budget`` above structural rather than
    lucky: the constraint is a mask over logits, so it holds for arbitrary weights.
    Which is also why it is run against the override-happy policy — against the
    trained one, which retains everywhere anyway, the mask is never consulted and
    the test would pass with ``budget_mask`` deleted.
    """
    from archetype_trader.phase3 import make_controller

    adapter = any_adapter

    ctrl = make_controller(
        adapter,
        grid=dataset.grid,
        n_archetypes=dataset.n_archetypes,
        budget=0,
        horizon=dataset.horizon,
        device=DEVICE,
    )
    grid = np.asarray(dataset.grid, dtype=np.float64)
    overrode = 0
    for row in range(min(16, len(dataset))):
        ctrl.reset(int(dataset.archetypes[row]))
        prev = int(np.argmin(np.abs(grid)))
        for t in range(dataset.horizon):
            base_action = int(dataset.base_actions[row, t])
            d = ctrl.decide(
                dataset.features[int(dataset.starts[row]) + t],
                base_action=base_action,
                position=float(grid[prev]),
                realized_cum=0.0,
                base_cum=0.0,
            )
            overrode += int(d.overrode)
            assert d.budget_left == 0, d.budget_left
            prev = d.action
    assert overrode == 0, f"{overrode} overrides taken with a budget of zero"


def test_the_budget_is_per_horizon_not_per_episode(dataset, adapter) -> None:
    """``reset`` must restore the full budget and clear the recurrent state.

    The adapter is a within-archetype device. Carrying budget or intent across a
    selector decision it did not make would let one horizon's exhausted budget
    silently disable adaptation for the rest of a run.
    """
    from archetype_trader.phase3 import make_controller

    ctrl = make_controller(
        adapter,
        grid=dataset.grid,
        n_archetypes=dataset.n_archetypes,
        budget=dataset.budget,
        horizon=dataset.horizon,
        device=DEVICE,
    )
    ctrl.reset(0)
    ctrl.budget_left = 0
    ctrl.in_override = True
    ctrl.step_index = dataset.horizon
    ctrl.reset(1)
    assert ctrl.budget_left == dataset.budget
    assert ctrl.in_override is False
    assert ctrl.step_index == 0
    assert float(ctrl.state.abs().max()) == 0.0, "recurrent state survived a reset"


# --------------------------------------------------------------------------- #
# Persistence and determinism
# --------------------------------------------------------------------------- #
def test_a_reloaded_adapter_scores_identically(cfg, dataset, phase3, any_adapter, tmp_path) -> None:
    """Save/load must not change a single decision.

    Checked through ``evaluate_adapter`` rather than by comparing state dicts:
    identical weights loaded into a differently-shaped module still answer
    differently, and the shape metadata (``n_actions``, ``n_archetypes``,
    ``hidden_dim``) travels in the checkpoint. ``load_adapter`` reads
    ``hidden_dim`` out of the stored ``phase3`` block, so dropping ``cfg=`` from
    the save is enough to silently rebuild a default-width network — which is
    exactly the failure this catches.

    Run against the override-happy policy as well as the trained one. The trained
    policy retains everywhere on this fixture, and a policy that retains everywhere
    scores ``improvement == 0.0`` and ``override_rate == 0.0`` no matter *what*
    width it was rebuilt at — so on the trained weights alone this test would agree
    with itself across a genuinely broken reload.
    """
    from dataclasses import replace

    from archetype_trader.phase3 import evaluate_adapter, load_adapter, save_adapter

    result = replace(phase3, model=any_adapter)
    rows = np.arange(min(64, len(dataset)))
    before = evaluate_adapter(cfg, result.model, dataset, rows, device=DEVICE)

    path = tmp_path / "adapter.pt"
    save_adapter(result, dataset, path, cfg=cfg)
    reloaded, blob = load_adapter(path, device=DEVICE)
    after = evaluate_adapter(cfg, reloaded, dataset, rows, device=DEVICE)

    assert blob["budget"] == dataset.budget
    assert tuple(blob["grid"]) == tuple(dataset.grid)
    assert blob["horizon"] == dataset.horizon
    assert blob["n_actions"] == len(dataset.grid)
    assert reloaded.hidden_dim == any_adapter.hidden_dim, (
        f"reloaded at width {reloaded.hidden_dim}, saved at {any_adapter.hidden_dim}: "
        "the checkpoint's phase3 block did not carry hidden_dim"
    )
    for key in ("improvement", "regret", "override_rate", "max_overrides", "target_accuracy"):
        assert after[key] == pytest.approx(before[key], rel=0, abs=1e-12), (
            key,
            before[key],
            after[key],
        )


def test_phase3_training_is_deterministic(bundle, dataset) -> None:
    """Same seed, same dataset: same curve, same selected checkpoint, same weights.

    The loss is compared bit-for-bit rather than approximately. A seeded run that
    agrees only to a tolerance has an unseeded source somewhere — most often a
    ``torch.Generator`` that was created but not threaded into the sampling call —
    and that is precisely the defect that makes a reported seed distribution
    unreproducible.
    """
    from archetype_trader.phase3 import train_phase3

    a = train_phase3(bundle.cfg, dataset, device=DEVICE, seed=7)
    b = train_phase3(bundle.cfg, dataset, device=DEVICE, seed=7)

    assert a.best_epoch == b.best_epoch, (a.best_epoch, b.best_epoch)
    assert a.best_val_improvement == b.best_val_improvement
    assert len(a.history) == len(b.history)
    for ha, hb in zip(a.history, b.history, strict=True):
        assert (ha.loss, ha.val_improvement, ha.override_rate) == (
            hb.loss,
            hb.val_improvement,
            hb.override_rate,
        ), f"epoch {ha.epoch} diverged"
    for (k, va), (_, vb) in zip(
        a.model.state_dict().items(), b.model.state_dict().items(), strict=True
    ):
        assert torch.equal(va, vb), f"parameter {k} differs between two seeded runs"


def test_early_stopping_keeps_the_best_epoch_not_the_last(cfg, bundle, dataset) -> None:
    """The returned weights must be the best-scoring checkpoint, not the final one.

    Worth its own test because of a reporting trap this stack walks into on
    low-signal data: the adapter learns to override *and* overriding loses money on
    validation, so the selected epoch is an early one that retains everywhere while
    the last epoch overrides constantly. Reading ``history[-1]``'s override rate
    next to ``best_val_improvement`` then describes a model that was discarded, and
    produces the nonsensical-looking pair "overrides on every horizon, improves by
    exactly zero". The two numbers must be read off the same epoch.
    """
    from archetype_trader.phase3 import evaluate_adapter, train_phase3

    res = train_phase3(bundle.cfg, dataset, device=DEVICE, seed=7)
    assert res.history, "no epochs ran"

    best = max(h.val_improvement for h in res.history)
    assert res.best_val_improvement == pytest.approx(best, rel=0, abs=1e-12), (
        f"best_val_improvement {res.best_val_improvement} is not the maximum over history "
        f"({best}); early stopping kept the wrong checkpoint"
    )
    assert res.history[res.best_epoch].val_improvement == pytest.approx(
        res.best_val_improvement, rel=0, abs=1e-12
    )
    # And the weights that came back really are that epoch's, not the last epoch's:
    # re-scoring the returned model on the same rows must reproduce the number.
    rng = np.random.default_rng(res.meta["seed"])
    _, val_rows = dataset.split(0.15, rng)
    if val_rows.size == 0:
        pytest.skip("no validation rows in this split")
    again = evaluate_adapter(cfg, res.model, dataset, val_rows, device=DEVICE)
    assert again["improvement"] == pytest.approx(res.meta["val_improvement"], rel=0, abs=1e-9), (
        again["improvement"],
        res.meta["val_improvement"],
    )
