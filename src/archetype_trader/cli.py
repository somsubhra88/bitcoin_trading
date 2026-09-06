"""Typer command line for the ArchetypeTrader stack.

Deliberately thin. Every command parses flags, builds an
:class:`~archetype_trader.config.ExperimentConfig`, and calls one function in
:mod:`archetype_trader.pipeline`. No logic lives here, so nothing that affects a
result can be reached only through the CLI and therefore only tested through it.

The commands are ordered as a pipeline::

    ingest -> features -> phase1 -> phase2 -> phase3 -> evaluate -> paper

``run`` chains the whole thing. ``-o key.path=value`` overrides any config field
(the value is YAML-decoded, so ``-o env.mode=perp`` and ``-o phase1.epochs=40``
both do the obvious thing).

Live trading has no command. ``paper --enable-live`` *requests* it and is refused
by :func:`archetype_trader.live.safety.preflight`, which ships an empty exchange
allowlist. That is the whole design: there is no code path here that can place a
real order, and enabling one is a commit that must also carry the adapter and its
tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from archetype_trader.config import ExperimentConfig, load_config, parse_override
from archetype_trader.utils.logging import configure_logging, get_logger

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ArchetypeTrader: three-phase archetype RL for BTC/USDT. Research code.",
)
logger = get_logger("archetype_trader.cli")

ConfigOpt = Annotated[
    Path,
    typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML experiment config."),
]
OverrideOpt = Annotated[
    list[str] | None,
    typer.Option("--override", "-o", help="Dotted config override, e.g. -o phase1.epochs=40."),
]
FoldOpt = Annotated[int, typer.Option("--fold", help="Walk-forward fold index.")]
SeedOpt = Annotated[
    int | None, typer.Option("--seed", help="Seed for this stage (defaults to config.seed).")
]
LogLevelOpt = Annotated[str, typer.Option("--log-level", help="DEBUG/INFO/WARNING/ERROR.")]
JsonLogsOpt = Annotated[bool, typer.Option("--json-logs", help="One JSON object per log line.")]


def _load(
    config: Path,
    override: list[str] | None,
    *,
    log_level: str = "INFO",
    json_logs: bool = False,
) -> ExperimentConfig:
    configure_logging(log_level, json_lines=json_logs)
    overrides: dict[str, Any] = dict(parse_override(t) for t in (override or []))
    cfg = load_config(config, overrides)
    logger.info("config %s (hash %s) -> %s", config, cfg.hash()[:12], cfg.run_dir())
    return cfg


def _echo_paths(paths: dict[str, Any]) -> None:
    for name, path in sorted(paths.items()):
        if path is not None:
            typer.echo(f"  {name}: {path}")


def _validate_ca_bundle(ca_bundle: Path | None) -> Path | None:
    if ca_bundle is not None and not ca_bundle.is_file():
        raise typer.BadParameter(
            f"CA bundle file not found: {ca_bundle}. "
            "Replace the placeholder with the path to your PEM certificate."
        )
    return ca_bundle


# --------------------------------------------------------------------------- #
@app.command()
def ingest(
    config: ConfigOpt,
    override: OverrideOpt = None,
    lenient: Annotated[
        bool,
        typer.Option(
            "--lenient",
            help="Downgrade data-quality errors to warnings. Use only to inspect bad data.",
        ),
    ] = False,
    log_level: LogLevelOpt = "INFO",
    json_logs: JsonLogsOpt = False,
) -> None:
    """Load, normalise and validate bars; write them plus a data-quality report."""
    from archetype_trader.pipeline import run_ingest

    cfg = _load(config, override, log_level=log_level, json_logs=json_logs)
    result = run_ingest(cfg, strict=not lenient)
    r = result.report
    typer.echo(
        f"{r.n_bars} bars {r.start} -> {r.end} ({r.symbol}, {r.bar_minutes}m, "
        f"{r.levels} book levels)"
    )
    typer.echo(
        f"duplicates removed={r.duplicates_removed} reordered={r.rows_reordered} "
        f"gaps={len(r.gaps)} missing_bars={r.n_gap_bars_missing} "
        f"zero_volume={r.zero_volume_bars} crossed_book={r.crossed_book_bars}"
    )
    for w in r.warnings:
        typer.secho(f"  warning: {w}", fg=typer.colors.YELLOW)
    for e in r.errors:
        typer.secho(f"  error: {e}", fg=typer.colors.RED)
    _echo_paths(result.paths)


@app.command()
def download(
    symbol: Annotated[str, typer.Option("--symbol", help="Exchange symbol, e.g. BTCUSDT.")],
    out: Annotated[Path, typer.Option("--out", help="Destination Parquet/CSV path.")],
    start: Annotated[str, typer.Option("--start", help="UTC start, YYYY-MM-DD.")],
    end: Annotated[str, typer.Option("--end", help="UTC end, YYYY-MM-DD (exclusive).")],
    interval: Annotated[
        str, typer.Option("--interval", help="Kline interval, e.g. 10m, 1h.")
    ] = "10m",
    base_url: Annotated[
        str, typer.Option("--base-url", help="Public REST base URL.")
    ] = "https://api.binance.com",
    ca_bundle: Annotated[
        Path | None,
        typer.Option(
            "--ca-bundle",
            help="PEM CA bundle for corporate/system TLS interception; verification stays enabled.",
        ),
    ] = None,
    log_level: LogLevelOpt = "INFO",
) -> None:
    """Fetch public historical klines. Public endpoints only - no credentials.

    Kept separate from ``ingest`` so a run cannot silently depend on the network:
    ``ingest`` reads a local file, and this is how that file gets there.
    """
    from archetype_trader.data.loaders import download_public_klines

    configure_logging(log_level)
    ca_bundle = _validate_ca_bundle(ca_bundle)
    path = download_public_klines(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        out_path=out,
        base_url=base_url,
        ca_bundle=ca_bundle,
    )
    typer.echo(f"wrote {path}")


@app.command()
def features(
    config: ConfigOpt,
    override: OverrideOpt = None,
    log_level: LogLevelOpt = "INFO",
    json_logs: JsonLogsOpt = False,
) -> None:
    """Build causal features, plan the purged walk-forward, fit per-fold scalers."""
    from archetype_trader.pipeline import run_features

    cfg = _load(config, override, log_level=log_level, json_logs=json_logs)
    result = run_features(cfg)
    f = result.features
    typer.echo(
        f"{f.n_features} features, valid_from={f.valid_from}, "
        f"max_lookback={f.max_lookback}, definition_hash={f.definition_hash[:12]}"
    )
    typer.echo(f"embargo={cfg.embargo_bars} bars (max_lookback + horizon + extra)")
    for i, fold in enumerate(result.plan.folds):
        typer.echo(
            f"  fold {i}: train={fold.train} val={fold.val} test={fold.test} "
            f"embargo={fold.embargo_bars}"
        )
    _echo_paths(result.paths)


@app.command()
def phase1(
    config: ConfigOpt,
    override: OverrideOpt = None,
    fold: FoldOpt = 0,
    seed: SeedOpt = None,
    log_level: LogLevelOpt = "INFO",
    json_logs: JsonLogsOpt = False,
) -> None:
    """Solve training-segment demonstrations; fit the VQ archetype codebook."""
    from archetype_trader.pipeline import run_phase1

    cfg = _load(config, override, log_level=log_level, json_logs=json_logs)
    out = run_phase1(cfg, fold_index=fold, seed=seed)
    d = out["diagnostics"]
    h, dist = d["health"], d["distinctness"]
    typer.echo(
        f"demos={d['n_demonstrations']} (last bar {d['last_demo_bar']} < val.start "
        f"{d['val_start']})"
    )
    typer.echo(
        f"codebook: utilization={h['utilization']:.2f} perplexity={h['perplexity']:.2f} "
        f"dead={len(h['dead_codes'])} collapsed={h['collapsed']}"
    )
    style = typer.colors.RED if dist["degenerate"] else typer.colors.GREEN
    typer.secho(
        f"behaviour: {dist['n_distinct_behaviours']}/{dist['codebook_size']} distinct, "
        f"code_gain={dist['code_gain']:+.4f} degenerate={dist['degenerate']}",
        fg=style,
    )
    typer.echo(f"  checkpoint: {out['checkpoint']}")


@app.command()
def phase2(
    config: ConfigOpt,
    override: OverrideOpt = None,
    fold: FoldOpt = 0,
    seed: SeedOpt = None,
    log_level: LogLevelOpt = "INFO",
    json_logs: JsonLogsOpt = False,
) -> None:
    """Train the recurrent archetype selector against the frozen Phase-1 decoder."""
    from archetype_trader.pipeline import run_phase2

    cfg = _load(config, override, log_level=log_level, json_logs=json_logs)
    out = run_phase2(cfg, fold_index=fold, seed=seed)
    d = out["diagnostics"]
    typer.echo(f"algo={d['algo']} best_objective={d['best_objective']:+.6f}")
    if d["history"]:
        last = d["history"][-1]
        typer.echo(
            f"final update: switch_rate={last.get('switch_rate', float('nan')):.3f} "
            f"cash_fraction={last.get('cash_fraction', float('nan')):.3f}"
        )
    typer.echo(f"  checkpoint: {out['checkpoint']}")


@app.command()
def phase3(
    config: ConfigOpt,
    override: OverrideOpt = None,
    fold: FoldOpt = 0,
    seed: SeedOpt = None,
    log_level: LogLevelOpt = "INFO",
    json_logs: JsonLogsOpt = False,
) -> None:
    """Train the budgeted step-level override adapter."""
    from archetype_trader.pipeline import run_phase3

    cfg = _load(config, override, log_level=log_level, json_logs=json_logs)
    out = run_phase3(cfg, fold_index=fold, seed=seed)
    d = out["diagnostics"]
    # Report the *selected* checkpoint, never the final epoch. Early stopping can
    # and does pick an early epoch, and the last epoch's override rate then
    # describes a model that was thrown away. Reading them side by side is how the
    # contradiction "overrides everywhere yet gains exactly zero" gets manufactured.
    m = d["meta"]
    dm = d["dataset_meta"]
    typer.echo(
        f"selected epoch {m['best_epoch']} of {cfg.phase3.epochs}: "
        f"val_improvement={m['best_val_improvement']:+.6f} log-return per horizon"
    )
    typer.echo(
        f"  val_override_rate={m['val_override_rate']:.3f} "
        f"max_overrides={m['val_max_overrides']:.2f}/{m['budget']} "
        f"regret_ratio={m['val_regret_ratio']:.3f} "
        f"(0 = captured all hindsight headroom, 1 = captured none)"
    )
    typer.echo(
        f"  hindsight headroom on train: mean={dm['mean_headroom']:+.4f} "
        f"median={dm['median_headroom']:+.4f} over {dm['n_episodes']} episodes"
    )
    if m["val_override_rate"] == 0.0:
        typer.secho(
            "  the selected adapter retains everywhere: the full stack is numerically "
            "identical to phase1_2, and the phase3 ablation row says nothing. Either "
            "the override is unlearnable on this data (a real negative result - check "
            "whether val_improvement went negative as the override rate rose) or "
            "phase3.epochs is too low.",
            fg=typer.colors.YELLOW,
        )
    if d["history"]:
        last = d["history"][-1]
        if last.get("val_override_rate", 0.0) != m["val_override_rate"]:
            typer.echo(
                f"  (final epoch reached val_override_rate="
                f"{last['val_override_rate']:.3f} with val_improvement="
                f"{last['val_improvement']:+.6f}; discarded by early stopping)"
            )
    typer.echo(f"  checkpoint: {out['checkpoint']}")


@app.command()
def evaluate(
    config: ConfigOpt,
    override: OverrideOpt = None,
    ablation: Annotated[
        list[str] | None,
        typer.Option("--ablation", help="Restrict to these ablation names (repeatable)."),
    ] = None,
    split: Annotated[
        list[str] | None, typer.Option("--split", help="Segments to score: val and/or test.")
    ] = None,
    seed: Annotated[
        list[int] | None, typer.Option("--seed", help="Override eval.seeds (repeatable).")
    ] = None,
    fold: Annotated[
        list[int] | None, typer.Option("--fold", help="Restrict to these folds (repeatable).")
    ] = None,
    no_baselines: Annotated[bool, typer.Option("--no-baselines")] = False,
    no_stress: Annotated[bool, typer.Option("--no-stress")] = False,
    no_bootstrap: Annotated[bool, typer.Option("--no-bootstrap")] = False,
    plots: Annotated[
        bool | None, typer.Option("--plots/--no-plots", help="Override eval.make_plots.")
    ] = None,
    log_level: LogLevelOpt = "INFO",
    json_logs: JsonLogsOpt = False,
) -> None:
    """Run the walk-forward matrix (variants, ablations, baselines, stress) and report.

    This retrains everything from the config and seeds. It does not read the
    checkpoints the phase commands wrote, so a reported number can never depend on
    a stale file.
    """
    from archetype_trader.pipeline import run_evaluate

    cfg = _load(config, override, log_level=log_level, json_logs=json_logs)
    out = run_evaluate(
        cfg,
        ablations=tuple(ablation) if ablation else None,
        splits=tuple(split) if split else ("val", "test"),
        seeds=tuple(seed) if seed else None,
        folds=tuple(fold) if fold else None,
        with_baselines=not no_baselines,
        with_stress=not no_stress,
        with_bootstrap=not no_bootstrap,
        make_plots=plots,
    )
    report = out["report"]
    verdict = report.verdict
    sel = report.selection
    typer.echo(f"{len(report.cells)} cells evaluated")
    chosen = sel.get("chosen")
    median = sel.get("median_by_strategy", {}).get(chosen)
    typer.echo(
        f"selected on val by {sel.get('metric')}: {chosen}"
        + ("" if median is None else f" (median {median:+.4f})")
    )
    accepted = bool(verdict.get("accepted"))
    n_criteria = len(verdict.get("criteria", []))
    typer.secho(
        f"verdict: {'ACCEPTED' if accepted else 'REJECTED as non-robust'} "
        f"({len(verdict.get('failures', []))}/{n_criteria} criteria failed)",
        fg=typer.colors.GREEN if accepted else typer.colors.RED,
        bold=True,
    )
    for f in verdict.get("failures", []):
        typer.secho(f"  fail: {f}", fg=typer.colors.RED)
    _echo_paths(out["paths"])


@app.command()
def paper(
    config: ConfigOpt,
    override: OverrideOpt = None,
    fold: FoldOpt = 0,
    seed: SeedOpt = None,
    split: Annotated[
        str, typer.Option("--split", help="Segment to replay: train, val or test.")
    ] = "test",
    max_bars: Annotated[int | None, typer.Option("--max-bars", help="Stop after N bars.")] = None,
    no_phase3: Annotated[bool, typer.Option("--no-phase3", help="Deploy Phase 1+2 only.")] = False,
    max_order_notional: Annotated[
        float, typer.Option("--max-order-notional", help="Operator cap per order.")
    ] = 0.0,
    max_position_notional: Annotated[
        float, typer.Option("--max-position-notional", help="Operator cap on position size.")
    ] = 0.0,
    max_leverage: Annotated[
        float, typer.Option("--max-leverage", help="Operator leverage ceiling.")
    ] = 1.0,
    enable_live: Annotated[
        bool,
        typer.Option(
            "--enable-live",
            help="REQUEST live execution. Refused: no exchange adapter ships here.",
        ),
    ] = False,
    exchange: Annotated[str, typer.Option("--exchange")] = "paper",
    mainnet: Annotated[
        bool, typer.Option("--mainnet", help="Ask for mainnet instead of testnet.")
    ] = False,
    confirm: Annotated[
        str | None, typer.Option("--confirm", help="Must be 'I ACCEPT THE RISK' for live.")
    ] = None,
    log_level: LogLevelOpt = "INFO",
    json_logs: JsonLogsOpt = False,
) -> None:
    """Replay a held-out segment through the deployment path (agent, shield, broker).

    Live execution is disabled by default and cannot be enabled by configuration.
    ``--enable-live`` prints a warning and asks the pre-flight gate, which refuses
    because no live exchange adapter exists in this repository.
    """
    from archetype_trader.live.safety import LiveLimits, SafetyError
    from archetype_trader.pipeline import run_paper

    cfg = _load(config, override, log_level=log_level, json_logs=json_logs)
    limits = LiveLimits(
        max_order_notional=max_order_notional,
        max_position_notional=max_position_notional,
        max_leverage=max_leverage,
        require_testnet=not mainnet,
    )
    try:
        out = run_paper(
            cfg,
            fold_index=fold,
            seed=seed,
            split=split,
            max_bars=max_bars,
            enable_live=enable_live,
            exchange=exchange,
            testnet=not mainnet,
            confirm_phrase=confirm,
            limits=limits,
            with_phase3=not no_phase3,
        )
    except SafetyError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=2) from exc

    res = out["result"]
    m = res.metrics
    typer.echo(
        f"{res.bars} bars, final equity {res.final_equity:,.2f} "
        f"({res.final_equity / cfg.env.initial_cash - 1.0:+.2%})"
    )
    typer.echo(
        f"orders={m.orders_submitted} skipped={m.orders_skipped} retries={m.submit_retries} "
        f"duplicates={m.duplicate_acks} failures={m.submit_failures}"
    )
    typer.echo(
        f"rejections={m.rejections} partial={m.partial_fills} "
        f"shield={m.shield_interventions} limits={m.limit_blocks} "
        f"forced_flats={m.forced_flats} liquidations={m.liquidations}"
    )
    style = typer.colors.YELLOW if m.reconciliation_breaches else typer.colors.GREEN
    typer.secho(
        f"reconciliation: {m.reconciliation_breaches} breaches, max drift {m.max_abs_drift:.3e}",
        fg=style,
    )
    typer.echo(f"agent: {json.dumps(res.agent_stats, default=str)}")
    _echo_paths(out["paths"])


@app.command()
def run(
    config: ConfigOpt,
    override: OverrideOpt = None,
    fold: FoldOpt = 0,
    seed: SeedOpt = None,
    skip_ingest: Annotated[
        bool, typer.Option("--skip-ingest", help="Reuse an already-ingested market.npz.")
    ] = False,
    skip_eval: Annotated[bool, typer.Option("--skip-eval")] = False,
    skip_paper: Annotated[bool, typer.Option("--skip-paper")] = False,
    log_level: LogLevelOpt = "INFO",
    json_logs: JsonLogsOpt = False,
) -> None:
    """Chain every stage: ingest, features, phase1-3, evaluate, paper."""
    from archetype_trader import pipeline

    cfg = _load(config, override, log_level=log_level, json_logs=json_logs)
    if not skip_ingest:
        typer.secho("[1/7] ingest", bold=True)
        pipeline.run_ingest(cfg)
    typer.secho("[2/7] features", bold=True)
    pipeline.run_features(cfg)
    for i, stage in enumerate((pipeline.run_phase1, pipeline.run_phase2, pipeline.run_phase3), 3):
        typer.secho(f"[{i}/7] phase{i - 2}", bold=True)
        stage(cfg, fold_index=fold, seed=seed)
    if not skip_eval:
        typer.secho("[6/7] evaluate", bold=True)
        out = pipeline.run_evaluate(cfg)
        v = out["report"].verdict
        typer.secho(
            f"verdict: {'ACCEPTED' if v.get('accepted') else 'REJECTED as non-robust'}",
            fg=typer.colors.GREEN if v.get("accepted") else typer.colors.RED,
            bold=True,
        )
        for f in v.get("failures", []):
            typer.secho(f"  fail: {f}", fg=typer.colors.RED)
    if not skip_paper:
        typer.secho("[7/7] paper", bold=True)
        res = pipeline.run_paper(cfg, fold_index=fold, seed=seed)["result"]
        typer.echo(f"paper: {res.bars} bars, final equity {res.final_equity:,.2f}")
    typer.echo(f"artifacts under {cfg.run_dir()}")


@app.command("show-config")
def show_config(
    config: ConfigOpt,
    override: OverrideOpt = None,
) -> None:
    """Print the fully resolved, validated config and its hash. No side effects."""
    import yaml

    cfg = _load(config, override, log_level="WARNING")
    typer.echo(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=True))
    typer.echo(f"# config_hash: {cfg.hash()}")
    typer.echo(f"# embargo_bars: {cfg.embargo_bars}")
    typer.echo(f"# action grid: {list(cfg.env.positions)}")
    typer.echo(f"# run_dir: {cfg.run_dir()}")
    if cfg.data.source == "synthetic":
        snr = cfg.data.synthetic.signal_to_noise
        typer.secho(
            f"# synthetic fixture: signal_to_noise={snr:.3f} "
            f"(drift sigma {cfg.data.synthetic.drift_sigma:.2e} / vol_low "
            f"{cfg.data.synthetic.vol_low:.2e}). Real 10-minute BTC is well below "
            "0.05; results from this fixture measure plumbing, not edge.",
            fg=typer.colors.YELLOW if snr > 0.2 else None,
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
