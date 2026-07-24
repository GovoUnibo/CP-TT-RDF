#!/usr/bin/env python3
"""Plot CP and TT parameter storage from the independent memory CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
LINK_NAME = "panda_link0"
CSV_PATH = SCRIPT_DIR / "result" / f"{LINK_NAME}_memory.csv"
POLYNOMIAL_ORDERS = (8, 16, 24)
COLORS = ("tab:blue", "tab:orange", "tab:green")


def load_rows() -> list[dict[str, object]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return [{"representation": row["representation"], "N": int(row["N"]), "rank": int(row["rank"]), "peak_gpu_gb": float(row["peak_gpu_gb"])} for row in csv.DictReader(handle)]


def save_plot(rows: list[dict[str, object]], representation: str) -> None:
    xlabel = "CP rank $R$" if representation == "cp" else "TT rank $r_1=r_2$"
    fig, ax = plt.subplots(figsize=(14.5, 8.6))
    for color, n in zip(COLORS, POLYNOMIAL_ORDERS):
        curve = sorted([row for row in rows if row["representation"] == representation and row["N"] == n], key=lambda row: int(row["rank"]))
        ax.plot([row["rank"] for row in curve], [row["peak_gpu_gb"] for row in curve], color=color, linewidth=3.2, label=f"N={n}")
        dense = next((row for row in rows if row["representation"] == "dense" and row["N"] == n), None)
        if dense:
            ax.axhline(dense["peak_gpu_gb"], color=color, linestyle="--", linewidth=3.2, alpha=0.8)
    ax.set_xlabel(xlabel, fontsize=23)
    ax.set_ylabel("Peak GPU memory [GB]", fontsize=23)
    ax.set_xlim(left=0.0)
    ax.tick_params(axis="both", which="both", labelsize=22)
    ax.grid(True, which="both", alpha=0.25)
    legend_handles = [
        Line2D([0], [0], color=color, linewidth=3.2, label=f"N={n}")
        for color, n in zip(COLORS, POLYNOMIAL_ORDERS)
    ]
    legend_handles.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=3.2, label="Dense reference")
    )
    ax.legend(handles=legend_handles, frameon=False, fontsize=18)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.14, top=0.88)
    output_path = SCRIPT_DIR / "result" / f"{LINK_NAME}_{representation}_memory.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {output_path}")


def main() -> None:
    rows = load_rows()
    save_plot(rows, "cp")
    save_plot(rows, "tt")


if __name__ == "__main__":
    main()
