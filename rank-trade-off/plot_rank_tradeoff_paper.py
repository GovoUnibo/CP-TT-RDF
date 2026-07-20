#!/usr/bin/env python3
"""Create publication-ready rank/order trade-off figures from sweep CSV data.

This script performs no training. It reads the CSV produced by
``single_link_rank_tradeoff.py`` and creates a compact figure suitable for the
paper and the reviewer response.

Typical usage from the repository root:

    python plot_rank_tradeoff_paper.py panda_link3

Optional explicit paths:

    python plot_rank_tradeoff_paper.py panda_link3 \
        --csv results/single_link_rank_tradeoff/panda_link3_metrics.csv \
        --output-dir results/single_link_rank_tradeoff/paper_figures

Outputs:

    <link>_rank_tradeoff_paper.pdf
    <link>_rank_tradeoff_paper.png
    <link>_pareto_tradeoff_paper.pdf
    <link>_pareto_tradeoff_paper.png

The main figure contains two rows (CP-RDF and TT-RDF) and three columns
(RMSE, inference runtime, and model storage). Each curve corresponds to one
Bernstein polynomial order N. The second figure shows the resulting error-cost
Pareto views.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


def discover_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir() and (candidate / "panda_test").is_dir():
            return candidate
    raise RuntimeError(f"Repository root not found from: {start}")


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = discover_repo_root(SCRIPT_DIR)
LINK_NAME = "panda_link0"
CSV_PATH = SCRIPT_DIR / "result" / f"{LINK_NAME}_metrics.csv"
RUNTIME_CSV_PATH = SCRIPT_DIR / "result" / f"{LINK_NAME}_runtime.csv"
OUTPUT_DIR = SCRIPT_DIR / "result"

# Same Matplotlib palette used by the benchmark plots.  Colour alone identifies
# the polynomial order, keeping the dense six-panel figure free of symbols.
ORDER_COLORS: Tuple[str, ...] = ("tab:blue", "tab:orange", "tab:green", "tab:red")

# Same parameters used in benchmark/alpha_beta_delta_model_plot.py.
TICK_FONT_SIZE = 22
LEGEND_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 23
TITLE_FONT_SIZE = 13
MEASURED_LINE_WIDTH = 3.2
PLOT_FIGSIZE = (14.5, 8.6)
DPI = 180

NUMERIC_FIELDS: Tuple[str, ...] = (
    "n_func",
    "rank_1",
    "rank_2",
    "rmse_mm",
    "runtime_median_ms",
    "runtime_std_ms",
    "model_storage_kib",
    "parameter_count",
    "dense_parameter_count",
    "compression_ratio_vs_dense",
)


# -----------------------------------------------------------------------------
# Input handling
# -----------------------------------------------------------------------------


def read_latest_successful_rows(csv_path: Path) -> List[Dict[str, Any]]:
    """Read the last successful row for each representation/order/rank tuple."""

    if not csv_path.is_file():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    latest: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "representation",
            "n_func",
            "rank_1",
            "rank_2",
            "rmse_mm",
            "runtime_median_ms",
            "runtime_std_ms",
            "model_storage_kib",
            "status",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"CSV {csv_path} is missing required columns: {sorted(missing)}"
            )

        for raw in reader:
            if raw.get("status", "").strip().lower() != "ok":
                continue
            try:
                row: Dict[str, Any] = dict(raw)
                for field in NUMERIC_FIELDS:
                    value = raw.get(field, "")
                    row[field] = float(value) if value not in (None, "") else math.nan

                representation = str(raw["representation"]).strip().lower()
                n_func = int(float(raw["n_func"]))
                rank_1 = int(float(raw["rank_1"]))
                rank_2 = int(float(raw["rank_2"]))
                row["representation"] = representation
                row["n_func"] = n_func
                row["rank_1"] = rank_1
                row["rank_2"] = rank_2
                latest[(representation, n_func, rank_1, rank_2)] = row
            except (KeyError, TypeError, ValueError):
                continue

    rows = list(latest.values())
    if not rows:
        raise RuntimeError(f"No successful configurations were found in {csv_path}")
    return rows


def merge_runtime_measurements(rows: List[Dict[str, Any]], runtime_csv_path: Path) -> None:
    """Replace legacy runtime values with the independent runtime benchmark."""

    if not runtime_csv_path.is_file():
        raise FileNotFoundError(f"Runtime CSV not found: {runtime_csv_path}")

    measured: Dict[Tuple[str, int, int], float] = {}
    with runtime_csv_path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                key = (str(raw["representation"]).lower(), int(raw["N"]), int(raw["rank"]))
                measured[key] = float(raw["runtime_ms"])
            except (KeyError, TypeError, ValueError):
                continue

    for row in rows:
        key = (str(row["representation"]), int(row["n_func"]), int(row["rank_1"]))
        if key in measured:
            row["runtime_median_ms"] = measured[key]


# -----------------------------------------------------------------------------
# Plot styling
# -----------------------------------------------------------------------------


def configure_matplotlib() -> None:
    """Use the same global output options as the benchmark plots."""

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def color_for_order(index: int) -> str:
    return ORDER_COLORS[index % len(ORDER_COLORS)]


def finite_positive(values: Iterable[float]) -> List[float]:
    return [float(value) for value in values if np.isfinite(value) and value > 0.0]


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
        ha="left",
    )


def apply_axis_style(ax: plt.Axes) -> None:
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(axis="both", which="both", labelsize=TICK_FONT_SIZE)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


# -----------------------------------------------------------------------------
# Main rank/order figure
# -----------------------------------------------------------------------------


def grouped_curve(
    rows: Sequence[Mapping[str, Any]], representation: str, order: int
) -> List[Mapping[str, Any]]:
    curve = [
        row
        for row in rows
        if row["representation"] == representation and int(row["n_func"]) == order
    ]
    return sorted(curve, key=lambda row: float(row["rank_1"]))


def dense_reference(
    rows: Sequence[Mapping[str, Any]], order: int, field: str
) -> float | None:
    """Return the dense Bernstein measurement for an order, when available."""

    for row in rows:
        if row["representation"] == "dense" and int(row["n_func"]) == order:
            value = float(row[field])
            return value if np.isfinite(value) else None
    return None


def plot_metric_curve(
    ax: plt.Axes,
    curve: Sequence[Mapping[str, Any]],
    field: str,
    color: str,
) -> None:
    ranks = np.asarray([float(row["rank_1"]) for row in curve], dtype=float)
    values = np.asarray([float(row[field]) for row in curve], dtype=float)
    valid = np.isfinite(ranks) & np.isfinite(values)
    ranks = ranks[valid]
    values = values[valid]
    if len(ranks) == 0:
        return

    # Plot only the measured medians: runtime standard-deviation bars are
    # intentionally omitted from the reviewer-facing figure.
    ax.plot(ranks, values, color=color, linestyle="-", linewidth=MEASURED_LINE_WIDTH)


def create_individual_figures(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, dpi: int
) -> List[Path]:
    """Create one benchmark-style PNG for each representation/metric pair."""

    orders = sorted({int(row["n_func"]) for row in rows})
    metric_specs = (
        ("rmse", "rmse_mm", "RMSE [mm]", False),
        ("runtime", "runtime_median_ms", "Inference time [ms]", False),
        ("storage", "model_storage_kib", "Model storage [KiB]", False),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []

    for representation in ("cp", "tt"):
        rank_label = "CP rank $R$" if representation == "cp" else "TT rank $r_1=r_2$"
        title_prefix = "CP-RDF" if representation == "cp" else "TT-RDF"
        for metric_name, field, ylabel, log_y in metric_specs:
            fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
            for order_index, order in enumerate(orders):
                curve = grouped_curve(rows, representation, order)
                if not curve:
                    continue
                color = color_for_order(order_index)
                plot_metric_curve(
                    ax,
                    curve,
                    field,
                    color,
                )
                dense_value = dense_reference(rows, order, field)
                if dense_value is not None:
                    ax.axhline(
                        dense_value,
                        color=color,
                        linestyle="--",
                        linewidth=MEASURED_LINE_WIDTH,
                        alpha=0.8,
                    )
                if log_y:
                    plotted_values = finite_positive(float(row[field]) for row in curve)
                    if plotted_values:
                        ax.set_yscale("log")

            ax.set_title(f"{title_prefix} — {ylabel}", fontsize=TITLE_FONT_SIZE)
            ax.set_xlabel(rank_label, fontsize=AXIS_LABEL_FONT_SIZE)
            ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
            apply_axis_style(ax)
            ax.legend(
                [f"N={order}" for order in orders if grouped_curve(rows, representation, order)],
                frameon=False,
                loc="best",
                fontsize=LEGEND_FONT_SIZE,
            )

            output_path = output_dir / f"{LINK_NAME}_{representation}_{metric_name}.png"
            fig.subplots_adjust(left=0.08, right=0.985, bottom=0.14, top=0.88)
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            output_paths.append(output_path)

    return output_paths


# -----------------------------------------------------------------------------
# Optional Pareto summary
# -----------------------------------------------------------------------------


def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return non-dominated points for minimization of both x and y."""

    valid = np.isfinite(x) & np.isfinite(y)
    mask = np.zeros(len(x), dtype=bool)
    valid_indices = np.flatnonzero(valid)
    for i in valid_indices:
        dominated = False
        for j in valid_indices:
            if i == j:
                continue
            no_worse = x[j] <= x[i] and y[j] <= y[i]
            strictly_better = x[j] < x[i] or y[j] < y[i]
            if no_worse and strictly_better:
                dominated = True
                break
        mask[i] = not dominated
    return mask


def create_pareto_figure(
    rows: Sequence[Mapping[str, Any]], output_stem: Path, dpi: int
) -> Tuple[Path, Path]:
    orders = sorted({int(row["n_func"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.20, top=0.79, wspace=0.32)

    x_specs = (
        ("runtime_median_ms", "Inference time [ms]"),
        ("model_storage_kib", "Model storage [KiB]"),
    )

    for ax_index, (x_field, x_label) in enumerate(x_specs):
        ax = axes[ax_index]
        for rep_index, representation in enumerate(("cp", "tt")):
            rep_rows = [row for row in rows if row["representation"] == representation]
            for order_index, order in enumerate(orders):
                curve = grouped_curve(rep_rows, representation, order)
                if not curve:
                    continue
                x = np.asarray([float(row[x_field]) for row in curve], dtype=float)
                y = np.asarray([float(row["rmse_mm"]) for row in curve], dtype=float)
                ranks = np.asarray([int(row["rank_1"]) for row in curve], dtype=int)
                valid = np.isfinite(x) & np.isfinite(y)
                x, y, ranks = x[valid], y[valid], ranks[valid]
                if len(x) == 0:
                    continue

                color = color_for_order(order_index)
                face = "none" if representation == "tt" else None
                label = f"{representation.upper()}, $N={order}$"
                ax.plot(
                    x,
                    y,
                    linestyle="none",
                    marker="o",
                    color=color,
                    markerfacecolor=face,
                    label=label,
                )

                frontier = pareto_mask(x, y)
                if np.any(frontier):
                    order_idx = np.argsort(x[frontier])
                    ax.plot(
                        x[frontier][order_idx],
                        y[frontier][order_idx],
                        linewidth=0.8,
                        alpha=0.65,
                    )

                for xi, yi, rank in zip(x, y, ranks):
                    ax.annotate(
                        str(rank),
                        (xi, yi),
                        xytext=(2.5, 2.5),
                        textcoords="offset points",
                        fontsize=5.5,
                        alpha=0.8,
                    )

        ax.set_xlabel(x_label)
        ax.set_ylabel("RMSE [mm]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        apply_axis_style(ax)
        add_panel_label(ax, "(a)" if ax_index == 0 else "(b)")

    axes[0].set_title("Accuracy--runtime trade-off")
    axes[1].set_title("Accuracy--storage trade-off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, max(1, len(handles))),
            frameon=False,
            bbox_to_anchor=(0.54, 0.99),
            columnspacing=1.0,
            handletextpad=0.35,
        )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    return pdf_path, png_path


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> int:
    configure_matplotlib()
    rows = read_latest_successful_rows(CSV_PATH)
    merge_runtime_measurements(rows, RUNTIME_CSV_PATH)

    representations = sorted({row["representation"] for row in rows})
    orders = sorted({int(row["n_func"]) for row in rows})
    print(f"Input CSV:        {CSV_PATH}")
    print(f"Runtime CSV:      {RUNTIME_CSV_PATH}")
    print(f"Successful rows: {len(rows)}")
    print(f"Representations: {representations}")
    print(f"Orders:          {orders}")

    for output_path in create_individual_figures(rows=rows, output_dir=OUTPUT_DIR, dpi=DPI):
        print(f"Plot:             {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
