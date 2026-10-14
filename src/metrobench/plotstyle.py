"""Shared Matplotlib style so that every figure of the project looks the same."""

import matplotlib as mpl

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAILURE = "#d03b3b"  # failure windows
GAP = "#c3c2b7"  # missing data
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def shade_failures(ax, failures, label: bool = True) -> None:
    """Shade the reported failure windows on a time axis."""
    for i, (start, end) in enumerate(failures[["start", "end"]].itertuples(index=False)):
        ax.axvspan(start, end, color=FAILURE, alpha=0.18, lw=0,
                   label="reported failure" if label and i == 0 else None)
