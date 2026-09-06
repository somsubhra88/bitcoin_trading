"""Typed YAML configuration for every stage.

Pydantic rather than Hydra: the repo already depends on pydantic, and a single
frozen, validated, hashable tree is easier to record in a run manifest than a
composed OmegaConf object. ``extra="forbid"`` everywhere turns a typo in a YAML
file into an error instead of a silently ignored key.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Mode = Literal["spot", "perp"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class SyntheticConfig(_Base):
    """Parameters for the synthetic market used by the smoke config and tests.

    This is a **plumbing fixture, not a testbed**, and the parameters are chosen
    to keep it from masquerading as one. See :attr:`signal_to_noise`: the AR(1)
    drift is deliberately a small fraction of the diffusion, because a fixture
    whose drift is *larger* than its noise lets a broken strategy look
    spectacular. An earlier default of ``drift_scale=4e-4`` gave a stationary
    drift standard deviation of 4.0e-3 against a low-regime bar volatility of
    1.5e-3 — a signal-to-noise ratio of 2.7, roughly two orders of magnitude
    above 10-minute BTC — and produced Sharpe ratios in the hundreds and 1000x
    price paths over a few thousand bars. Every number that came out of it was an
    artifact of the generator.
    """

    n_bars: int = Field(default=20_000, gt=100)
    start: str = "2021-01-01T00:00:00Z"
    seed: int = 7
    initial_price: float = Field(default=30_000.0, gt=0)
    # Regime-switching GBM: two vol states with sticky transitions, plus a
    # slow mean-reverting drift so trend-following has something to find.
    vol_low: float = Field(default=0.0015, gt=0)
    vol_high: float = Field(default=0.0060, gt=0)
    regime_persistence: float = Field(default=0.995, gt=0, lt=1)
    drift_ar: float = Field(default=0.995, ge=0, lt=1)
    #: Per-bar innovation of the AR(1) drift. 8e-6 with ``drift_ar=0.995`` gives a
    #: stationary drift sigma of 8.0e-5, i.e. ~5% of ``vol_low`` — small enough
    #: that a strategy has to be right about something, large enough that the
    #: learned components are not fitting pure noise.
    drift_scale: float = Field(default=8e-6, ge=0)
    spread_bps_low: float = Field(default=1.0, ge=0)
    spread_bps_high: float = Field(default=6.0, ge=0)

    @property
    def drift_sigma(self) -> float:
        """Stationary standard deviation of the AR(1) drift, per bar."""
        return float(self.drift_scale / math.sqrt(max(1.0 - self.drift_ar**2, 1e-12)))

    @property
    def signal_to_noise(self) -> float:
        """Predictable drift sigma over low-regime diffusion sigma.

        Reported so the artifact is visible rather than buried. Real 10-minute BTC
        sits well below 0.05; anything approaching 1.0 means the fixture, not the
        strategy, is producing the result.
        """
        return self.drift_sigma / self.vol_low


class DataConfig(_Base):
    source: Literal["synthetic", "csv", "parquet"] = "synthetic"
    path: str | None = None
    symbol: str = "BTCUSDT"
    bar_minutes: int = Field(default=10, gt=0)
    #: 0 disables order-book features (OHLCV-only baseline). 25 is the target
    #: depth for the full model.
    orderbook_levels: int = Field(default=0, ge=0, le=100)
    max_gap_bars: int = Field(
        default=6, ge=1, description="Gaps beyond this are reported as breaks."
    )
    synthetic: SyntheticConfig = SyntheticConfig()

    @model_validator(mode="after")
    def _path_required(self) -> DataConfig:
        if self.source in ("csv", "parquet") and not self.path:
            raise ValueError(f"data.path is required for source={self.source!r}")
        return self


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
class FeatureConfig(_Base):
    lookback: int = Field(default=48, gt=1, description="Bars in the observation window.")
    return_windows: tuple[int, ...] = (1, 3, 6, 12, 24, 72)
    vol_windows: tuple[int, ...] = (12, 36, 144)
    ma_windows: tuple[int, ...] = (12, 36, 144)
    rsi_window: int = Field(default=14, gt=1)
    atr_window: int = Field(default=14, gt=1)
    macd: tuple[int, int, int] = (12, 26, 9)
    momentum_windows: tuple[int, ...] = (6, 24, 72)
    downside_vol_window: int = Field(default=36, gt=1)
    orderbook_depths: tuple[int, ...] = (1, 5, 10, 25)
    include_time_encodings: bool = True
    include_orderbook: bool = False
    clip_sigma: float = Field(default=8.0, gt=0, description="Post-scaling winsorisation.")

    @property
    def max_lookback(self) -> int:
        """Longest history any feature touches — drives the split embargo."""
        return max(
            self.lookback,
            max(self.return_windows),
            max(self.vol_windows),
            max(self.ma_windows),
            max(self.momentum_windows),
            self.rsi_window,
            self.atr_window,
            sum(self.macd[:2]),
            self.downside_vol_window,
        )


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
class SplitConfig(_Base):
    scheme: Literal["rolling", "anchored"] = "rolling"
    n_folds: int = Field(default=3, ge=1)
    train_bars: int = Field(default=8_000, gt=100)
    val_bars: int = Field(default=2_000, gt=10)
    test_bars: int = Field(default=2_000, gt=10)
    #: Extra embargo on top of ``max_lookback + horizon`` (which is enforced).
    extra_embargo_bars: int = Field(default=0, ge=0)


# --------------------------------------------------------------------------- #
# Execution / environment
# --------------------------------------------------------------------------- #
class CostConfig(_Base):
    taker_fee_bps: float = Field(default=5.0, ge=0)
    maker_fee_bps: float = Field(default=1.0, ge=0)
    #: Fraction of order flow assumed to rest as maker; the remainder is taker.
    maker_fill_ratio: float = Field(default=0.0, ge=0, le=1)
    #: Half-spread paid when no order book is available, in bps of mid.
    fallback_half_spread_bps: float = Field(default=1.0, ge=0)
    #: Square-root market impact, ``slip_frac = coef * sigma_bar * sqrt(Q/V)``,
    #: where ``sigma_bar`` is trailing per-bar return volatility and ``Q/V`` is
    #: participation. ``coef`` is the dimensionless Y of the square-root law
    #: (empirically O(1)); a raw bps coefficient would be volatility-blind and
    #: mis-price the same order across regimes.
    slippage_coef: float = Field(default=1.0, ge=0)
    #: Trailing window for the impact volatility estimate.
    slippage_vol_window: int = Field(default=36, gt=1)
    #: Floor so a zero-volatility stretch still charges something.
    slippage_floor_bps: float = Field(default=0.1, ge=0)
    #: Cap on the fraction of a bar's volume the strategy may consume.
    max_participation: float = Field(default=0.05, gt=0, le=1)
    #: Bars of decision-to-execution delay (>=1 enforces next-bar execution).
    latency_bars: int = Field(default=1, ge=1)
    min_notional: float = Field(default=10.0, ge=0)
    min_qty: float = Field(default=1e-5, ge=0)
    funding_bps_per_8h: float = Field(default=1.0)
    funding_interval_bars: int = Field(default=48, gt=0)
    walk_order_book: bool = True
    #: Bars without a fresh observation before data are declared stale.
    stale_after_bars: int = Field(default=2, ge=1)


class RewardConfig(_Base):
    turnover_penalty: float = Field(default=0.0, ge=0)
    downside_penalty: float = Field(default=0.0, ge=0)
    drawdown_penalty: float = Field(default=0.0, ge=0)
    inventory_penalty: float = Field(default=0.0, ge=0)
    vol_target_annual: float | None = Field(default=None, gt=0)
    scale: float = Field(default=1.0, gt=0)


class EnvConfig(_Base):
    mode: Mode = "spot"
    initial_cash: float = Field(default=100_000.0, gt=0)
    leverage: float = Field(default=1.0, gt=0)
    #: Maintenance-margin fraction used for the perp liquidation check.
    maintenance_margin: float = Field(default=0.005, gt=0, lt=1)
    max_episode_bars: int = Field(default=2_000, gt=1)
    #: Levels per side in the target-position grid. ``1`` gives the {0,+1} /
    #: {-1,0,+1} grids the three phases use; larger values give the finer grid the
    #: volatility-targeted and continuous-position (SAC) baselines need, since
    #: scaling exposure by inverse volatility is meaningless on three levels.
    position_levels: int = Field(default=1, ge=1)
    costs: CostConfig = CostConfig()
    reward: RewardConfig = RewardConfig()

    @property
    def positions(self) -> tuple[float, ...]:
        """Discrete target positions available in this mode."""
        n = self.position_levels
        longs = tuple(i / n for i in range(1, n + 1))
        if self.mode == "spot":
            return (0.0, *longs)
        return (*(-x for x in reversed(longs)), 0.0, *longs)


# --------------------------------------------------------------------------- #
# Risk shield
# --------------------------------------------------------------------------- #
class ShieldConfig(_Base):
    enabled: bool = True
    max_gross_exposure: float = Field(default=1.0, gt=0)
    max_net_exposure: float = Field(default=1.0, gt=0)
    max_leverage: float = Field(default=1.0, gt=0)
    vol_target_annual: float | None = Field(default=None, gt=0)
    vol_lookback: int = Field(default=36, gt=1)
    max_daily_loss: float = Field(default=0.05, gt=0)
    max_drawdown: float = Field(default=0.25, gt=0)
    cooldown_bars: int = Field(default=6, ge=0)
    loss_streak_trigger: int = Field(default=3, ge=1)
    max_spread_bps: float = Field(default=25.0, gt=0)
    min_bar_volume: float = Field(default=0.0, ge=0)
    max_est_slippage_bps: float = Field(default=40.0, gt=0)
    min_liquidation_distance: float = Field(default=0.15, gt=0)
    kill_switch: bool = False


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #
class PlannerConfig(_Base):
    horizon: int = Field(default=72, gt=1)
    #: 1 reproduces the paper's "at most one position change"; 2 is the
    #: entry/exit ablation.
    max_changes: int = Field(default=1, ge=1, le=8)
    cost_bps: float | None = Field(
        default=None,
        description="Round-trip-aware per-unit cost; defaults to taker fee + half spread.",
    )


class Phase1Config(_Base):
    n_trajectories: int = Field(default=30_000, gt=1)
    horizon: int = Field(default=72, gt=1)
    codebook_size: int = Field(default=10, gt=1)
    code_dim: int = Field(default=16, gt=1)
    hidden_dim: int = Field(default=128, gt=1)
    encoder: Literal["gru", "lstm"] = "gru"
    layers: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=256, gt=0)
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    grad_clip: float = Field(default=1.0, gt=0)
    commitment_beta: float = Field(default=0.25, gt=0)
    codebook_loss_weight: float = Field(default=1.0, ge=0)
    ema_codebook: bool = True
    ema_decay: float = Field(default=0.99, gt=0, lt=1)
    dead_code_restart_after: int = Field(
        default=3, ge=1, description="Epochs of zero use before reinit."
    )
    early_stopping_patience: int = Field(default=5, ge=1)
    amp: bool = False
    teacher_forcing: float = Field(default=1.0, ge=0, le=1)
    min_perplexity: float = Field(default=2.0, gt=0, description="Collapse alarm threshold.")


class Phase2Config(_Base):
    algo: Literal["recurrent_ppo", "recurrent_dqn"] = "recurrent_ppo"
    hidden_dim: int = Field(default=128, gt=1)
    #: Bars of pre-horizon context the selector may look at.
    context_bars: int = Field(default=48, gt=1)
    updates: int = Field(default=60, ge=1)
    horizons_per_update: int = Field(default=64, ge=1)
    epochs_per_update: int = Field(default=4, ge=1)
    minibatch_size: int = Field(default=16, ge=1)
    lr: float = Field(default=3e-4, gt=0)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, gt=0, le=1)
    clip_range: float = Field(default=0.2, gt=0)
    entropy_coef: float = Field(default=0.01, ge=0)
    value_coef: float = Field(default=0.5, ge=0)
    grad_clip: float = Field(default=0.5, gt=0)
    #: Auxiliary CE toward the Phase-1 demonstration's archetype label.
    aux_label_coef: float = Field(default=0.1, ge=0)
    include_cash_archetype: bool = True
    finetune_decoder: bool = False
    reward_scale: float = Field(default=100.0, gt=0)
    eps_start: float = Field(default=1.0, gt=0, le=1, description="recurrent_dqn only.")
    eps_end: float = Field(default=0.05, gt=0, le=1)
    target_sync_every: int = Field(default=5, ge=1)


class Phase3Config(_Base):
    hidden_dim: int = Field(default=64, gt=1)
    override_budget: int = Field(default=1, ge=1)
    #: 40, not 10. Measured on the synthetic fold: 8 epochs captured 1.7% of the
    #: hindsight-override headroom and 40 captured 31%. Below ~20 the adapter
    #: retains the base plan everywhere, which makes the Phase-3 row numerically
    #: identical to Phase-1+2 and the ablation uninformative.
    epochs: int = Field(default=40, ge=1)
    batch_size: int = Field(default=128, gt=0)
    lr: float = Field(default=3e-4, gt=0)
    grad_clip: float = Field(default=1.0, gt=0)
    n_episodes: int = Field(default=2_000, gt=1)
    regret_coef: float = Field(default=1.0, ge=0)
    imitation_coef: float = Field(default=0.5, ge=0)
    cost_coef: float = Field(default=1.0, ge=0)
    turnover_penalty: float = Field(default=0.0, ge=0)
    drawdown_penalty: float = Field(default=0.0, ge=0)
    downside_penalty: float = Field(default=0.0, ge=0)
    entropy_coef: float = Field(default=0.01, ge=0)


class BaselineConfig(_Base):
    enabled: tuple[str, ...] = (
        "cash",
        "buy_and_hold",
        "ema_cross",
        "macd",
        "vol_target_trend",
        "supervised_threshold",
        "recurrent_ppo",
        "dqn",
        "sac_continuous",
    )
    ema_fast: int = Field(default=12, gt=1)
    ema_slow: int = Field(default=72, gt=1)
    #: Annualised volatility the vol-target trend baseline sizes toward. Its
    #: exposure is capped at 1.0, so this only ever scales *down*.
    vol_target_annual: float = Field(default=0.5, gt=0)
    threshold_quantile: float = Field(default=0.6, gt=0.5, lt=1)
    supervised_epochs: int = Field(default=10, ge=1)
    rl_updates: int = Field(default=30, ge=1)
    rl_steps_per_update: int = Field(default=1_024, ge=8)
    rl_hidden: int = Field(default=128, gt=1)


class EvalConfig(_Base):
    bars_per_year: int = Field(default=52_560, gt=0, description="10-minute bars in 365 days.")
    cvar_alpha: float = Field(default=0.05, gt=0, lt=0.5)
    bootstrap_samples: int = Field(default=1_000, ge=10)
    bootstrap_block_bars: int = Field(default=144, ge=1)
    stress_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    #: Predeclared model-selection objective. Never computed on test.
    selection_metric: Literal["calmar", "sharpe", "composite"] = "calmar"
    make_plots: bool = True


class ExperimentConfig(_Base):
    name: str = "smoke"
    seed: int = 0
    seeds: tuple[int, ...] = (0,)
    device: str = "auto"
    output_dir: str = "runs"
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    splits: SplitConfig = SplitConfig()
    env: EnvConfig = EnvConfig()
    shield: ShieldConfig = ShieldConfig()
    planner: PlannerConfig = PlannerConfig()
    phase1: Phase1Config = Phase1Config()
    phase2: Phase2Config = Phase2Config()
    phase3: Phase3Config = Phase3Config()
    baselines: BaselineConfig = BaselineConfig()
    eval: EvalConfig = EvalConfig()

    @model_validator(mode="after")
    def _cross_checks(self) -> ExperimentConfig:
        if self.phase1.horizon != self.planner.horizon:
            raise ValueError(
                "phase1.horizon must equal planner.horizon "
                f"({self.phase1.horizon} != {self.planner.horizon})"
            )
        if self.features.include_orderbook and self.data.orderbook_levels == 0:
            raise ValueError("features.include_orderbook requires data.orderbook_levels > 0")
        if self.data.orderbook_levels and max(self.features.orderbook_depths) > (
            self.data.orderbook_levels
        ):
            raise ValueError(
                "features.orderbook_depths exceeds data.orderbook_levels "
                f"({max(self.features.orderbook_depths)} > {self.data.orderbook_levels})"
            )
        if self.env.mode == "spot" and self.env.leverage != 1.0:
            raise ValueError("spot mode requires env.leverage == 1.0")
        return self

    @property
    def embargo_bars(self) -> int:
        """Mandatory purge/embargo: longest lookback + trading horizon."""
        return self.features.max_lookback + self.planner.horizon + self.splits.extra_embargo_bars

    def hash(self) -> str:
        from archetype_trader.utils.provenance import sha256_json

        return sha256_json(self.model_dump(mode="json"))

    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load a YAML config, apply dotted overrides, validate.

    ``overrides`` keys use dots (``"env.mode": "perp"``) so the CLI can accept
    ``-o env.mode=perp`` without knowing the schema.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"config root must be a mapping, got {type(raw).__name__}")
    for dotted, value in (overrides or {}).items():
        _assign_dotted(raw, dotted, value)
    return ExperimentConfig.model_validate(raw)


def _assign_dotted(tree: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = tree
    for key in parts[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``key.path=value``; the value is YAML-decoded so types survive."""
    if "=" not in text:
        raise ValueError(f"override must look like key=value, got {text!r}")
    key, _, raw = text.partition("=")
    return key.strip(), yaml.safe_load(raw)


def dump_config(cfg: ExperimentConfig, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=True))
    return p
