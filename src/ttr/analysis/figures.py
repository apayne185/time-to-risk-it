"""Figures for the EDA / replication report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from ttr.analysis.replication import DFAResult
from ttr.viz import LABELS, MUTED, SERIES, TEXT_SECONDARY, apply_style, save

PANELS = [
    ("active_share", "Share of players who bet", "{:.0%}"),
    ("bet_days_per_player", "Betting days per player", "{:.0f}"),
    ("sports_bets_per_day", "Sportsbook bets per betting day (median)", "{:.0f}"),
]


def trajectories_figure(traj: pd.DataFrame, path: Path) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(12, 3.6), sharex=True)
    for ax, (col, title, fmt) in zip(axes, PANELS, strict=True):
        for group in ("control", "case"):
            g = traj[traj["group"] == group].sort_values("month")
            ax.plot(g["month"], g[col], color=SERIES[group], label=LABELS[group])
            last = g.iloc[-1]
            ax.annotate(
                LABELS[group],
                (last["month"], last[col]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=TEXT_SECONDARY,
            )
        ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, f=fmt: f.format(v)))
        ax.set_xlabel("Months relative to RG intervention")
    axes[0].annotate(
        "intervention",
        (0, 0),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Harm-onset cases bet more often than controls, and escalate before intervention",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.legend(
        handles[::-1], labels[::-1], loc="upper right", ncols=2, fontsize=9, bbox_to_anchor=(1, 1.0)
    )
    return save(fig, path)


def structure_figure(results: list[DFAResult], path: Path, top: int = 10) -> Path:
    """Dot plot: correlation of each index with the discriminant score, both analyses."""
    apply_style()
    by_name = {r.name: r for r in results}
    full, pre = by_name["full_history"].structure, by_name["pre_index"].structure
    names = list(full.index[:top])[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y = range(len(names))
    for i, n in enumerate(names):
        ax.plot([full[n], pre[n]], [i, i], color="#d6d5cf", linewidth=2, zorder=1)
    ax.scatter(
        [full[n] for n in names],
        y,
        s=64,
        facecolor="white",
        edgecolor=MUTED,
        linewidth=2,
        zorder=2,
        label=f"Whole history (AUC {by_name['full_history'].cv_auc:.2f})",
    )
    ax.scatter(
        [pre[n] for n in names],
        y,
        s=64,
        color=SERIES["control"],
        edgecolor="white",
        linewidth=2,
        zorder=3,
        label=f"Before intervention only (AUC {by_name['pre_index'].cv_auc:.2f})",
    )
    ax.set_yticks(list(y), [n.replace("_", " ") for n in names])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.3, 0.85)
    ax.set_xlabel("Correlation with discriminant score (structure coefficient)")
    ax.set_title("Live-action intensity separates cases from controls,\neven before intervention")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return save(fig, path)
