"""Shared figure style. Colours are the validated reference palette (light mode).

Cases are always orange and controls always blue, in every figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
MUTED = "#a3a29c"

SERIES = {"control": "#2a78d6", "case": "#eb6834"}  # categorical slots 1 and 2
LABELS = {"case": "Harm-onset RG cases", "control": "Matched controls"}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": TEXT,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": TEXT,
            "axes.titlelocation": "left",
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
        }
    )


def save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
