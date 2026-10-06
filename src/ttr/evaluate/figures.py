"""Decision-layer figures. Decision model in blue; rule baseline as the muted reference."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, PercentFormatter

from ttr.evaluate.run import EvaluationResult
from ttr.viz import GRID, MUTED, SERIES, TEXT_SECONDARY, apply_style, save

DECISION = SERIES["control"]  # categorical slot 1
PCT = FuncFormatter(lambda v, _: f"{v * 100:.3g}%")

NAMES = {
    "rule_baseline": "Rule baseline",
    "cox": "Cox",
    "xgb_cox": "XGBoost Cox",
    "xgb_aft": "XGBoost AFT",
    "torch_hazard": "PyTorch hazard net",
}


def _label_end(ax: Axes, x: float, y: float, text: str) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=TEXT_SECONDARY,
    )


def capacity_figure(result: EvaluationResult, path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for family, color, width in (
        (result.decision_family, DECISION, 2.4),
        ("rule_baseline", MUTED, 2.0),
    ):
        cap = result.evaluations[family].capacity
        ax.plot(
            cap["share"],
            cap["recall"],
            color=color,
            linewidth=width,
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=NAMES[family],
        )
        _label_end(ax, cap["share"].iloc[-1], cap["recall"].iloc[-1], NAMES[family])
    ax.set_xscale("log")
    ax.set_xticks(result.evaluations["rule_baseline"].capacity["share"])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1%}".replace(".0%", "%")))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Share of active players contacted each month")
    ax.set_ylabel("Next-month RG cases reached")
    ax.set_title("Contacting the riskiest players reaches far more upcoming RG cases")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return save(fig, path)


def decision_curve_figure(result: EvaluationResult, path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    t = result.thresholds
    scale = 10_000  # net benefit per 10,000 active players
    ax.axhline(0, color=GRID, linewidth=1.5)
    ax.plot(
        t,
        result.contact_all * scale,
        color=MUTED,
        linestyle=(0, (4, 3)),
        linewidth=1.5,
        label="Contact everyone",
    )
    for family, color in ((result.decision_family, DECISION), ("rule_baseline", MUTED)):
        nb = result.evaluations[family].net_benefit * scale
        ax.plot(t, nb, color=color, label=NAMES[family])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(PCT)
    top = result.evaluations[result.decision_family].net_benefit.max() * scale
    ax.set_ylim(-0.25 * top, top * 1.2)
    ax.set_xlabel("Risk threshold for contact (30-day probability, log scale)")
    ax.set_ylabel("Net benefit per 10,000 players")
    ax.set_title("The model adds value over contacting everyone or no one")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return save(fig, path)


def calibration_figure(result: EvaluationResult, path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    static, rolling = result.reliability, result.reliability_rolling
    lo = max(min(static["predicted"].min(), rolling["predicted"].min()) * 0.7, 1e-6)
    hi = max(static["observed"].max(), rolling["predicted"].max()) * 1.5
    ax.plot([lo, hi], [lo, hi], color=GRID, linewidth=1.5, zorder=1)
    ax.scatter(
        static["predicted"],
        static["observed"].clip(lower=lo),
        s=48,
        facecolor="white",
        edgecolor=MUTED,
        linewidth=2,
        zorder=2,
        label="Static calibration",
    )
    ax.scatter(
        rolling["predicted"],
        rolling["observed"].clip(lower=lo),
        s=48,
        color=DECISION,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
        label="Monthly intercept update",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.grid(axis="x")
    ax.set_xlabel("Predicted 30-day risk")
    ax.set_ylabel("Observed 30-day rate")
    ax.set_title("Monthly updates move predictions toward observed rates")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return save(fig, path)


def importance_figure(result: EvaluationResult, path: Path, top: int = 12) -> Path:
    apply_style()
    imp = result.importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(np.arange(len(imp)), imp.to_numpy(), height=0.6, color=DECISION)
    ax.set_yticks(np.arange(len(imp)), [n.replace("_", " ") for n in imp.index])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")
    ax.set_xlabel("Mean absolute contribution to the risk score (test set)")
    ax.set_title(f"What drives the {NAMES[result.decision_family]} risk score")
    fig.tight_layout()
    return save(fig, path)
