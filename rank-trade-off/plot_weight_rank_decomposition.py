#!/usr/bin/env python3
"""Create CP and TT weight-compression plots from the saved CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
LINK_NAME = "panda_link0"
CSV_PATH = SCRIPT_DIR / "result" / f"{LINK_NAME}_weight_decomposition.csv"
CP_PNG_PATH = SCRIPT_DIR / "result" / f"{LINK_NAME}_cp_weight_decomposition.png"
TT_PNG_PATH = SCRIPT_DIR / "result" / f"{LINK_NAME}_tt_weight_decomposition.png"
POLYNOMIAL_ORDERS = (8, 16, 24)
COLORS = ("tab:blue", "tab:orange", "tab:green")


def load_rows() -> list[dict[str, object]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return [
            {
                "representation": row["representation"],
                "N": int(row["N"]),
                "rank": int(row["rank"]),
                "rmse_weights": float(row["rmse_weights"]),
            }
            for row in csv.DictReader(handle)
        ]


def save_plot(rows: list[dict[str, object]], representation: str, output_path: Path) -> None:
    xlabel = "CP rank $R$" if representation == "cp" else "TT rank $r_1=r_2$"
    fig, ax = plt.subplots(figsize=(14.5, 8.6))
    for index, n_func in enumerate(POLYNOMIAL_ORDERS):
        curve = sorted(
            [row for row in rows if row["representation"] == representation and row["N"] == n_func],
            key=lambda row: int(row["rank"]),
        )
        if curve:
            ax.plot(
                [row["rank"] for row in curve],
                [row["rmse_weights"] for row in curve],
                color=COLORS[index],
                linewidth=3.2,
                label=f"N={n_func}",
            )
    ax.set_xlabel(xlabel, fontsize=23)
    ax.set_ylabel("Weight-tensor RMSE", fontsize=23)
    ax.set_xlim(left=0.0)
    ax.tick_params(axis="both", which="both", labelsize=22)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=18)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.14, top=0.88)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    save_plot(rows, "cp", CP_PNG_PATH)
    save_plot(rows, "tt", TT_PNG_PATH)
    print(f"CP PNG: {CP_PNG_PATH}")
    print(f"TT PNG: {TT_PNG_PATH}")


if __name__ == "__main__":
    main()
