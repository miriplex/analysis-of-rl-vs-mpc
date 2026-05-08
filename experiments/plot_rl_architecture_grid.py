from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "figures" / "rl_architectures"
PNG_PATH = OUT_DIR / "rl_architecture_grid_mlp_clean.png"
PDF_PATH = OUT_DIR / "rl_architecture_grid_mlp_clean.pdf"


@dataclass(frozen=True)
class MLPTopology:
    title: str
    hidden_layers: tuple[int, ...]
    study_scope: str


TOPOLOGIES: tuple[MLPTopology, ...] = (
    MLPTopology("MLP 16-16", (16, 16), "First-order + pendulum"),
    MLPTopology("MLP 32-32", (32, 32), "First-order + pendulum"),
    MLPTopology("MLP 16-16-16", (16, 16, 16), "First-order only"),
    MLPTopology("MLP 32-32-32", (32, 32, 32), "First-order + pendulum"),
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


def _arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.8,
            color="#949494",
            zorder=2,
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
            boxstyle="round,pad=0.012,rounding_size=0.03",
            linewidth=1.8,
            edgecolor=ec,
            facecolor=fc,
            zorder=3,
        )
    )
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=fs, weight=weight, color=tc, zorder=4)


def _layer_badge(ax, x: float) -> None:
    _rounded_box(
        ax,
        x - 0.055,
        0.71,
        0.11,
        0.06,
        "tanh",
        fc="#ffe6c9",
        ec="#d78733",
        tc="#8a4d17",
        fs=12.0,
    )


def _draw_layer(ax, x: float, width: int) -> None:
    radius = 0.033
    circle_ys = [0.60, 0.49, 0.34, 0.23]
    for y in circle_ys:
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                linewidth=1.8,
                edgecolor="#2e769d",
                facecolor="#d7edf8",
                zorder=3,
            )
        )
    ax.text(x, 0.415, "⋮", ha="center", va="center", fontsize=24, color="#6d6d6d", zorder=4)
    ax.text(x, 0.12, f"{width}", ha="center", va="center", fontsize=20, weight="bold", color="#245170")
    ax.text(x, 0.065, "neurons", ha="center", va="center", fontsize=12.5, color="#5e5e5e")


def _draw_output(ax, x: float) -> None:
    ax.add_patch(
        Circle(
            (x, 0.44),
            0.040,
            linewidth=1.8,
            edgecolor="#3d6827",
            facecolor="#d8edcc",
            zorder=3,
        )
    )
    ax.text(x, 0.44, "u", ha="center", va="center", fontsize=16, weight="bold", color="#2d4c1f", zorder=4)


def _draw_topology(ax, topo: MLPTopology) -> None:
    _panel(ax)
    ax.text(0.07, 0.89, topo.title, ha="left", va="top", fontsize=23, weight="bold", color="#202020")
    ax.text(0.07, 0.81, topo.study_scope, ha="left", va="top", fontsize=14, color="#666666")

    _rounded_box(
        ax,
        0.08,
        0.31,
        0.12,
        0.18,
        "Input",
        fc="#f5ecd9",
        ec="#b79a63",
        tc="#564221",
        fs=18,
    )

    n_hidden = len(topo.hidden_layers)
    layer_xs = [0.52, 0.72] if n_hidden == 2 else [0.46, 0.62, 0.78]

    _arrow(ax, 0.21, 0.44, layer_xs[0] - 0.06, 0.44)
    for i, (x, width) in enumerate(zip(layer_xs, topo.hidden_layers)):
        _layer_badge(ax, x)
        _draw_layer(ax, x, width)
        if i > 0:
            _arrow(ax, layer_xs[i - 1] + 0.05, 0.44, x - 0.05, 0.44)

    _arrow(ax, layer_xs[-1] + 0.05, 0.44, 0.89, 0.44)
    _draw_output(ax, 0.93)


def render() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))
    for ax, topo in zip(axes.flat, TOPOLOGIES):
        _draw_topology(ax, topo)

    fig.suptitle("MLP RL controller architectures used in the study", fontsize=22, weight="bold", y=0.985)
    fig.text(
        0.5,
        0.035,
        "Hidden layers use tanh activation. In the main benchmark runs the final output is linear: u = u_raw.",
        ha="center",
        fontsize=13,
        color="#505050",
    )

    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.95))
    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    render()
