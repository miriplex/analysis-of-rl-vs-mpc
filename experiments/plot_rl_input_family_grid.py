from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "figures" / "rl_architectures"
PNG_PATH = OUT_DIR / "rl_input_family_grid.png"
PDF_PATH = OUT_DIR / "rl_input_family_grid.pdf"


@dataclass(frozen=True)
class FeatureBlock:
    text: str


@dataclass(frozen=True)
class InputFamily:
    title: str
    subtitle: str
    blocks: tuple[FeatureBlock, ...]


FAMILIES: tuple[InputFamily, ...] = (
    InputFamily(
        title="Compact6",
        subtitle="First-order MLPs",
        blocks=(
            FeatureBlock("current reference / output / error"),
            FeatureBlock("error integral and slope"),
            FeatureBlock("previous control"),
        ),
    ),
    InputFamily(
        title="Rich11",
        subtitle="First-order MLPs",
        blocks=(
            FeatureBlock("current reference / output / error"),
            FeatureBlock("previous outputs and errors"),
            FeatureBlock("error integral and slope"),
            FeatureBlock("previous two controls"),
        ),
    ),
    InputFamily(
        title="State6",
        subtitle="Pendulum MLPs",
        blocks=(
            FeatureBlock("current x, x_dot, theta, theta_dot errors"),
            FeatureBlock("accumulated angle error"),
            FeatureBlock("previous control"),
        ),
    ),
    InputFamily(
        title="Rich11",
        subtitle="Pendulum MLPs",
        blocks=(
            FeatureBlock("current x, x_dot, theta, theta_dot errors"),
            FeatureBlock("previous x, x_dot, theta, theta_dot errors"),
            FeatureBlock("accumulated angle error"),
            FeatureBlock("previous control and increment"),
        ),
    ),
)


def _panel(ax) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.03, 0.03),
            0.94,
            0.93,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            linewidth=1.4,
            edgecolor="#d7d7d7",
            facecolor="#fcfcfc",
            zorder=0,
        )
    )


def _rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str,
    ec: str,
    tc: str,
    fs: float,
    weight: str = "bold",
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.5,
            edgecolor=ec,
            facecolor=fc,
            zorder=2,
        )
    )
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=fs, weight=weight, color=tc, zorder=3)


def _arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.8,
            color="#9a9a9a",
            zorder=2,
        )
    )


def _draw_block(ax, x: float, y: float, w: float, h: float, block: FeatureBlock) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.5,
            edgecolor="#6fa3c5",
            facecolor="#edf4fb",
            zorder=2,
        )
    )
    ax.text(x + 0.02, y + h / 2.0, block.text, ha="left", va="center", fontsize=10.5, color="#355d78", zorder=3)


def _draw_family(ax, family: InputFamily) -> None:
    _panel(ax)
    ax.text(0.07, 0.89, family.title, ha="left", va="top", fontsize=24, weight="bold", color="#202020")
    ax.text(0.07, 0.81, family.subtitle, ha="left", va="top", fontsize=14, color="#666666")

    if len(family.blocks) == 3:
        block_positions = ((0.07, 0.59), (0.07, 0.40), (0.07, 0.21))
        block_size = (0.39, 0.12)
    else:
        block_positions = ((0.07, 0.64), (0.07, 0.49), (0.07, 0.34), (0.07, 0.19))
        block_size = (0.41, 0.092)

    for (x, y), block in zip(block_positions, family.blocks):
        _draw_block(ax, x, y, block_size[0], block_size[1], block)

    ax.text(0.595, 0.62, "feature\nvector", ha="center", va="center", fontsize=14, color="#6a6a6a")
    _arrow(ax, 0.47, 0.47, 0.68, 0.47)
    _rounded_box(
        ax,
        0.69,
        0.41,
        0.15,
        0.18,
        "MLP",
        fc="#f5ecd9",
        ec="#b79a63",
        tc="#564221",
        fs=22,
    )
    ax.text(0.765, 0.64, "same\nbackbones", ha="center", va="bottom", fontsize=12.5, color="#7a6a4b")
    _arrow(ax, 0.84, 0.50, 0.91, 0.50)
    _rounded_box(
        ax,
        0.92,
        0.45,
        0.04,
        0.08,
        "u",
        fc="#d8edcc",
        ec="#3d6827",
        tc="#2d4c1f",
        fs=16,
    )


def render() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.0))
    for ax, family in zip(axes.flat, FAMILIES):
        _draw_family(ax, family)

    fig.suptitle("MLP input families used across the study", fontsize=24, weight="bold", y=0.985)
    fig.text(
        0.5,
        0.04,
        "Compact6 and Rich11 were used in the first-order study. State6 and a pendulum-specific Rich11 variant were used on the inverted pendulum.",
        ha="center",
        fontsize=12.8,
        color="#505050",
    )

    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.95))
    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    render()
