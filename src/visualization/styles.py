from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": True,
        }
    )


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(
    fig,
    output_dir: str | Path,
    name: str,
    *,
    dpi: int = 300,
    vector_format: str = "pdf",
) -> None:

    out = ensure_dir(output_dir)
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out / f"{name}.{vector_format}", bbox_inches="tight")
