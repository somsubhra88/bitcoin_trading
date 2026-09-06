"""Turn a :class:`~archetype_trader.eval.harness.WalkForwardReport` into artifacts.

Writes, into one run directory:

``report.md``
    The human-readable write-up. The verdict is the **first** section, before any
    table. A report that buries "this strategy is not robust" under twelve pages of
    metrics is a marketing document.
``metrics.csv`` / ``report.json``
    The same numbers, machine-readable. The CSV is one row per (fold, seed,
    strategy, split) — never pre-aggregated, so a reader can recompute any median
    the report quotes.
``series.npz``
    Representative equity and return paths.
``plots/*.png``
    Optional; skipped without matplotlib.

Nothing here computes a metric. Every number written is copied from the report,
which got it from :mod:`archetype_trader.eval.metrics`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from archetype_trader.eval import plots as plotting
from archetype_trader.eval.ablations import ABLATIONS_BY_NAME, REFERENCE
from archetype_trader.eval.harness import WalkForwardReport, report_to_json
from archetype_trader.utils.logging import get_logger

logger = get_logger(__name__)

DISCLAIMER = (
    "This is a research artifact. Backtested and simulated results are not evidence "
    "of future performance, no return is promised or implied, and nothing here is "
    "financial advice."
)

_TABLE_COLUMNS = (
    ("strategy", "strategy", "s"),
    ("total_return", "total ret", "pct"),
    # The log-annualised column, not the compounded one: compounding a three-day
    # test segment to a year is arithmetically valid and completely unreadable.
    ("annualized_log_return", "ann log ret", "f2"),
    ("annualized_vol", "ann vol", "pct"),
    ("sharpe", "Sharpe", "f2"),
    ("sortino", "Sortino", "f2"),
    ("calmar", "Calmar", "f2"),
    ("max_drawdown", "maxDD", "pct"),
    ("cvar", "CVaR5%", "pct4"),
    ("turnover", "turnover", "f1"),
    ("fees", "fees", "f0"),
    ("funding", "funding", "f0"),
    ("n_trades", "trades", "d"),
    ("hit_rate", "hit", "pct"),
    ("exposure", "expo", "pct"),
    ("max_leverage", "maxLev", "f2"),
    ("n_liquidations", "liq", "d"),
    ("shield_intervention_rate", "shield", "pct"),
    ("final_equity", "final eq", "f0"),
)


def _fmt(value: Any, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "s":
        return str(value)
    v = float(value)
    if not np.isfinite(v):
        return "nan"
    return {
        "pct": f"{v * 100:.2f}%",
        "pct4": f"{v * 100:.4f}%",
        "f0": f"{v:,.0f}",
        "f1": f"{v:.1f}",
        "f2": f"{v:.2f}",
        "d": f"{int(v)}",
    }[kind]


def _md_table(rows: list[dict[str, Any]], columns: tuple[tuple[str, str, str], ...]) -> str:
    if not rows:
        return "_(no rows)_\n"
    head = "| " + " | ".join(label for _, label, _ in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(_fmt(r.get(key), kind) for key, _, kind in columns) + " |" for r in rows
    ]
    return "\n".join([head, rule, *body]) + "\n"


def _median_rows(report: WalkForwardReport, split: str) -> list[dict[str, Any]]:
    """One row per strategy: the **median** across (fold, seed), never the best.

    Medians per column, computed independently. That means the row is not any single
    realised run, which is stated in the report text rather than hidden — the
    alternative (pick the run closest to the median Sharpe) hides variance in every
    other column.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for cell in report.cells:
        if cell.split == split:
            grouped.setdefault(cell.strategy, []).append(cell.row())
    out: list[dict[str, Any]] = []
    for strategy, rows in grouped.items():
        agg: dict[str, Any] = {"strategy": strategy, "n_runs": len(rows)}
        for key, _, _ in _TABLE_COLUMNS:
            if key == "strategy":
                continue
            vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float, bool))]
            agg[key] = float(np.median(vals)) if vals else None
        out.append(agg)
    return sorted(out, key=lambda r: (-(r["calmar"] or -1e18), -(r["sharpe"] or -1e18)))


def _verdict_section(report: WalkForwardReport) -> str:
    v = report.verdict
    lines: list[str] = ["## Verdict", ""]
    if v.get("accepted"):
        lines += [
            f"**`{REFERENCE}` passed all {len(v.get('criteria', []))} predeclared robustness "
            "criteria on the test folds.** That is a statement about this data, this cost "
            "model and these seeds, and nothing more.",
        ]
    else:
        lines += [
            f"**REJECTED as non-robust.** `{REFERENCE}` failed "
            f"{len(v.get('failures', []))} of {len(v.get('criteria', []))} predeclared "
            "criteria:",
            "",
        ]
        lines += [f"{i}. {f}" for i, f in enumerate(v.get("failures", []), 1)]
    lines += ["", "Criteria, fixed in code before any result was seen:", ""]
    lines += [f"- {c}" for c in v.get("criteria", [])]
    checks = v.get("checks", {})
    if checks:
        lines += ["", "<details><summary>Measured values</summary>", "", "```json"]
        lines += [json.dumps(_jsonable(checks), indent=2)]
        lines += ["```", "", "</details>"]
    return "\n".join(lines) + "\n"


def _selection_section(report: WalkForwardReport) -> str:
    sel = report.selection
    lines = [
        "## Model selection",
        "",
        f"Predeclared objective: **{sel.get('metric')}**, taken as the median across "
        "seeds and folds on the *validation* segments only.",
        "",
        f"Chosen configuration: **`{sel.get('chosen')}`**.",
        "",
    ]
    medians = sel.get("median_by_strategy", {})
    if medians:
        lines += ["| configuration | validation median | n |", "|---|---|---|"]
        for name, value in sorted(medians.items(), key=lambda kv: -kv[1]):
            n = sel.get("n_observations", {}).get(name, 0)
            lines.append(f"| {name} | {value:.4f} | {n} |")
        lines.append("")
    lines += [
        "Test metrics played no part in this choice. The harness computes them after "
        "selection has already returned.",
        "",
    ]
    return "\n".join(lines)


def _ablation_section(report: WalkForwardReport) -> str:
    rows = _median_rows(report, "test")
    by_name = {r["strategy"]: r for r in rows}
    ref = by_name.get(REFERENCE)
    lines = ["## Ablations", ""]
    if ref is None:
        return "\n".join([*lines, "_(the reference strategy produced no test rows)_", ""])
    lines += [
        "Deltas are median test Sharpe relative to `full`. A **positive** delta means "
        "removing that component helped, which is a finding about the design, not a bug "
        "in the ablation.",
        "",
        "| ablation | isolates | Sharpe | ΔSharpe | Calmar | maxDD |",
        "|---|---|---|---|---|---|",
    ]
    for name, ab in ABLATIONS_BY_NAME.items():
        row = by_name.get(name)
        if row is None:
            continue
        delta = (row["sharpe"] or 0.0) - (ref["sharpe"] or 0.0)
        # A vacuous row has ΔSharpe of exactly 0.00 for an uninteresting reason.
        # Marking it inline is the difference between "this knob does not matter"
        # and "this knob was never turned".
        mark = " ⚠︎ vacuous" if name in report.vacuous else ""
        lines.append(
            f"| `{name}`{mark} | {ab.isolates} | {_fmt(row['sharpe'], 'f2')} | {delta:+.2f} | "
            f"{_fmt(row['calmar'], 'f2')} | {_fmt(row['max_drawdown'], 'pct')} |"
        )
    if report.vacuous:
        lines += [
            "",
            "⚠︎ **Vacuous ablations.** The patch for these rows sets a field to the "
            "value it already held in this config, so the stack trained is identical "
            f"to `{REFERENCE}` and the row duplicates it exactly. A ΔSharpe of 0.00 "
            "here says nothing about the component. To make the comparison real, "
            "enable the field in the base config first.",
            "",
        ]
        for name, keys in sorted(report.vacuous.items()):
            lines.append(f"- `{name}`: already at the patched value: {', '.join(keys)}")
    return "\n".join(lines) + "\n"


def _interval_section(report: WalkForwardReport) -> str:
    if not report.intervals:
        return ""
    lines = [
        "## Bootstrap confidence intervals (test)",
        "",
        "Stationary bootstrap on the net per-bar log-return path, so serial dependence "
        "is preserved. `excludes_zero=false` means the metric is not distinguishable "
        "from zero at this level.",
        "",
        "| strategy | metric | point | lo | hi | excludes zero |",
        "|---|---|---|---|---|---|",
    ]
    for strategy, ivs in report.intervals.items():
        for metric, iv in ivs.items():
            lines.append(
                f"| {strategy} | {metric} | {iv.point:.4f} | {iv.lo:.4f} | {iv.hi:.4f} | "
                f"{str(iv.excludes_zero).lower()} |"
            )
    return "\n".join(lines) + "\n"


def _stress_section(report: WalkForwardReport, initial_equity: float) -> str:
    ref = report.stress.get(REFERENCE)
    if not ref:
        return ""
    lines = [
        "## Friction stress test",
        "",
        "Fees, half-spread, slippage coefficient and funding are all multiplied "
        "together. Latency is *ceiled*, never relaxed below one bar, and maximum "
        "participation is divided — so a higher multiplier is strictly harder to "
        "trade, in every dimension at once.",
        "",
        f"Fold {ref.get('fold')}, seed {ref.get('seed')}, split `{ref.get('split')}`.",
        "",
        "| multiplier | final equity | net PnL | trades | fees | funding | maxDD |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in ref.get("points", []):
        pnl = p["final_equity"] - initial_equity
        lines.append(
            f"| {p['multiplier']:g}x | {p['final_equity']:,.0f} | {pnl:+,.0f} | "
            f"{p['n_trades']} | {p['fees']:,.0f} | {p['funding']:+,.0f} | "
            f"{p['max_drawdown'] * 100:.2f}% |"
        )
    survives = ref.get("survives_friction")
    lines += [
        "",
        f"Survives friction up to {ref.get('tolerated_multiplier', 2.0)}x: "
        f"**{str(bool(survives)).lower()}**.",
    ]
    for reason in ref.get("reasons", []):
        lines.append(f"- {reason}")
    return "\n".join(lines) + "\n"


def _seed_section(report: WalkForwardReport) -> str:
    stats = {k: v for k, v in report.seed_stats.items() if k.endswith("|test")}
    if not stats:
        return ""
    lines = [
        "## Seed distribution (test)",
        "",
        "The median column is the headline number everywhere in this report. The max "
        "column is printed only so a reader can see how much better cherry-picking "
        "would have looked.",
        "",
        "| strategy | seeds | median Sharpe | q25 | q75 | min | max | frac ret > 0 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    nan = float("nan")
    for key, row in sorted(stats.items(), key=lambda kv: -kv[1].get("sharpe_median", -1e18)):
        strategy = key.split("|")[0]
        lines.append(
            f"| {strategy} | {int(row.get('sharpe_n_seeds', 0))} | "
            f"{row.get('sharpe_median', nan):.3f} | {row.get('sharpe_q25', nan):.3f} | "
            f"{row.get('sharpe_q75', nan):.3f} | {row.get('sharpe_min', nan):.3f} | "
            f"{row.get('sharpe_max', nan):.3f} | "
            f"{row.get('total_return_frac_positive', nan):.2f} |"
        )
    return "\n".join(lines) + "\n"


def _regime_section(report: WalkForwardReport) -> str:
    table = report.regimes.get("test_sharpe_by_regime", {})
    if not table:
        return ""
    regimes = sorted({r for row in table.values() for r in row})
    lines = [
        "## Conditional on volatility regime (test Sharpe)",
        "",
        "Regime terciles are fitted on the *training* segment and applied unchanged, so "
        '"high volatility" means the same thing in every column. These are per-bar '
        "statistics over a gappy subseries, not tradable paths.",
        "",
        "| strategy | " + " | ".join(regimes) + " |",
        "|---|" + "|".join("---" for _ in regimes) + "|",
    ]
    for strategy, row in table.items():
        lines.append(
            "| "
            + strategy
            + " | "
            + " | ".join(f"{row.get(r, float('nan')):.2f}" for r in regimes)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _attribution_section(report: WalkForwardReport) -> str:
    ref = [c for c in report.cells if c.split == "test" and c.strategy == REFERENCE]
    if not ref:
        return ""
    cell = ref[0]
    lines = [
        "## Cost attribution (reference, first test run)",
        "",
        "| component | quote currency |",
        "|---|---|",
    ]
    for key in ("gross_pnl", "fees", "funding", "execution_cost", "net_pnl", "residual"):
        if key in cell.attribution:
            lines.append(f"| {key} | {cell.attribution[key]:+,.2f} |")
    lines += [
        "",
        f"The residual is the accounting check: gross minus fees, funding and execution "
        f"must equal the realised change in equity. Here it is "
        f"{cell.attribution.get('residual', 0.0):+.3e}.",
    ]
    return "\n".join(lines) + "\n"


def _limitations_section(report: WalkForwardReport) -> str:
    prov = report.provenance
    n_test = len({(c.fold, c.seed) for c in report.cells if c.split == "test"})
    return "\n".join(
        [
            "## Known limitations",
            "",
            f"- Evaluated over {n_test} (fold, seed) test combination(s). rl.md asks for "
            "at least 10 seeds; anything less is a smoke test, not evidence.",
            "- Fills are simulated. Spread, square-root impact and participation caps are "
            "*models*, and a real book can be worse than all three at once.",
            "- Funding is applied at the configured interval from a configured rate, not "
            "from the exchange's realised funding history unless one was ingested.",
            "- The Phase-1 planner sees the future by construction. It runs on training "
            "segments only; the inference-path import test is what enforces that, and it "
            "is a test, not a proof.",
            "- No slippage from being the marginal participant in a crash, no exchange "
            "downtime, no API rate limiting, no margin-call mechanics beyond the "
            "configured liquidation check.",
            f"- Data source: `{prov.get('data_source', 'unspecified')}`. Synthetic data "
            "cannot discriminate between methods; it only proves the plumbing runs.",
            "",
            DISCLAIMER,
            "",
        ]
    )


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def render_markdown(report: WalkForwardReport, *, initial_equity: float, title: str) -> str:
    """Assemble the Markdown report. Verdict first, tables after."""
    prov = report.provenance
    parts = [
        f"# {title}",
        "",
        f"> {DISCLAIMER}",
        "",
        "| | |",
        "|---|---|",
        f"| config hash | `{report.config_hash}` |",
        f"| git commit | `{prov.get('git_commit', 'unknown')}` |",
        f"| data hash | `{prov.get('data_hash', 'unknown')}` |",
        f"| feature hash | `{prov.get('feature_hash', 'unknown')}` |",
        f"| generated | {prov.get('timestamp', 'unknown')} |",
        f"| torch | {prov.get('torch_version', 'unknown')} |",
        f"| python | {prov.get('python_version', 'unknown')} |",
        f"| device | {prov.get('device', 'unknown')} |",
        "",
        _verdict_section(report),
        _selection_section(report),
        "## Test results (median across folds and seeds)",
        "",
        "Every row is a per-column median, so the row is not any single realised run. "
        "Per-run numbers are in `metrics.csv`.",
        "",
        _md_table(_median_rows(report, "test"), _TABLE_COLUMNS),
        "",
        "## Validation results (median across folds and seeds)",
        "",
        _md_table(_median_rows(report, "val"), _TABLE_COLUMNS),
        "",
        _ablation_section(report),
        _interval_section(report),
        _seed_section(report),
        _stress_section(report, initial_equity),
        _regime_section(report),
        _attribution_section(report),
        _limitations_section(report),
    ]
    return "\n".join(p for p in parts if p is not None)


def write_csv(report: WalkForwardReport, path: Path) -> Path:
    rows = [c.row() for c in report.cells]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_plots(report: WalkForwardReport, out_dir: Path, *, initial_equity: float) -> list[Path]:
    """Best-effort plotting. A missing matplotlib is a warning, never a failure."""
    if not plotting.HAVE_MPL:
        logger.warning("matplotlib unavailable; skipping plots")
        return []
    made: list[Path] = []
    test_eq = {k.split("|")[0]: v for k, v in report.equity_curves.items() if k.endswith("|test")}
    test_r = {k.split("|")[0]: v for k, v in report.return_series.items() if k.endswith("|test")}
    if test_eq:
        made.append(plotting.plot_equity(test_eq, out_dir / "equity.png"))
        made.append(plotting.plot_underwater(test_eq, out_dir / "underwater.png"))
    if test_r:
        window = min(720, max(20, min(len(v) for v in test_r.values()) // 4))
        made.append(plotting.plot_rolling_risk(test_r, out_dir / "rolling_risk.png", window=window))
    ref_cells = [c for c in report.cells if c.split == "test" and c.strategy == REFERENCE]
    if ref_cells:
        cell = ref_cells[0]
        s = cell.summary
        if "proposed_positions" in s and "positions" in s:
            made.append(
                plotting.plot_positions(
                    np.asarray(s["proposed_positions"], dtype=np.float64),
                    np.asarray(s["positions"], dtype=np.float64),
                    out_dir / "positions.png",
                    name=REFERENCE,
                )
            )
        if "gross_log_returns" in s:
            made.append(
                plotting.plot_gross_vs_net(
                    np.cumsum(np.asarray(s["gross_log_returns"], dtype=np.float64)),
                    np.cumsum(np.asarray(s["net_log_returns"], dtype=np.float64)),
                    out_dir / "gross_vs_net.png",
                    name=REFERENCE,
                )
            )
        made.append(plotting.plot_cost_attribution(cell.attribution, out_dir / "costs.png"))
        usage = cell.metrics.extra.get("archetype_usage")
        if usage is not None:
            made.append(
                plotting.plot_archetype_usage(
                    np.asarray(usage, dtype=np.float64), out_dir / "archetype_usage.png"
                )
            )
    sharpes = report.by_strategy("test", "sharpe")
    if sharpes and max(len(v) for v in sharpes.values()) > 1:
        made.append(
            plotting.plot_seed_distribution(sharpes, out_dir / "seeds.png", metric="sharpe")
        )
    ref_stress = report.stress.get(REFERENCE)
    if ref_stress:
        made.append(
            plotting.plot_stress(
                {p["multiplier"]: p["final_equity"] for p in ref_stress["points"]},
                out_dir / "stress.png",
                initial_equity=initial_equity,
            )
        )
    table = report.regimes.get("test_sharpe_by_regime")
    if table:
        made.append(plotting.plot_regime_conditioned(table, out_dir / "regimes.png"))
    return made


def write_report(
    report: WalkForwardReport,
    out_dir: str | Path,
    *,
    initial_equity: float,
    title: str = "ArchetypeTrader walk-forward evaluation",
    make_plots: bool = True,
) -> dict[str, Path]:
    """Write every artifact and return the paths, so the CLI can print them."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    md = out / "report.md"
    md.write_text(render_markdown(report, initial_equity=initial_equity, title=title), "utf-8")
    paths["report_md"] = md

    js = out / "report.json"
    js.write_text(json.dumps(_jsonable(report_to_json(report)), indent=2), "utf-8")
    paths["report_json"] = js

    paths["metrics_csv"] = write_csv(report, out / "metrics.csv")

    if report.equity_curves or report.return_series:
        npz = out / "series.npz"
        np.savez_compressed(
            npz,
            **{f"equity::{k}": v for k, v in report.equity_curves.items()},
            **{f"returns::{k}": v for k, v in report.return_series.items()},
        )
        paths["series_npz"] = npz

    if make_plots:
        made = write_plots(report, out / "plots", initial_equity=initial_equity)
        for p in made:
            paths[f"plot_{p.stem}"] = p
    return paths
