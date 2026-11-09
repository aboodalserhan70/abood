"""
توليد أشكال توضيحية للتقرير: مخطط خط الأنابيب ومخطط بنية النموذج.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import FIGURES_DIR

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 9})

BLUE, GREEN, ORANGE, GRAY, RED = "#2E86C1", "#27AE60", "#D68910", "#7F8C8D", "#C0392B"


def _box(ax, x, y, w, h, text, color, text_color="white", fontsize=8.5, radius=0.02):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=1.1, edgecolor="#1B2631", facecolor=color, alpha=0.95, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight="bold", zorder=4,
            linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2, color="#34495E", linestyle="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
        linewidth=1.4, color=color, zorder=2, linestyle=linestyle,
        shrinkA=0.5, shrinkB=0.5))


def pipeline_diagram(out: Path | None = None) -> Path:
    """مخطط خط الأنابيب: صفّان مع موصّلات بشكل زاوية قائمة (elbow)."""
    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    row1 = [("1  Dataset\n4 classes", BLUE),
            ("2  Pre-processing\nresize 224×224", BLUE),
            ("3  Stratified split\n70 / 15 / 15", GRAY),
            ("4  Augmentation\nflip · zoom · colour", ORANGE)]
    row2 = [("5  CNN model\nscratch / MobileNetV2", GREEN),
            ("6  Training\nAdamW · class weights", GREEN),
            ("7  Evaluation\nacc · P/R/F1 · κ", RED),
            ("8  Grad-CAM\nexplainability", RED)]

    w, h, gap, x0 = 2.55, 1.05, 0.42, 0.32
    y1, y2 = 3.15, 1.45
    xs = [x0 + i * (w + gap) for i in range(4)]

    for i, (label, colour) in enumerate(row1):
        _box(ax, xs[i], y1, w, h, label, colour, fontsize=8.6)
        if i:
            _arrow(ax, xs[i] - gap + 0.04, y1 + h / 2, xs[i] - 0.02, y1 + h / 2)

    # موصّل على شكل زاوية قائمة: نهاية الصف الأول -> بداية الصف الثاني
    y_mid = (y1 + y2 + h) / 2
    ax.plot([xs[3] + w / 2, xs[3] + w / 2, xs[0] + w / 2],
            [y1, y_mid, y_mid], color="#34495E", lw=1.5, zorder=1)
    _arrow(ax, xs[0] + w / 2, y_mid, xs[0] + w / 2, y2 + h + 0.02)

    for i, (label, colour) in enumerate(row2):
        _box(ax, xs[i], y2, w, h, label, colour, fontsize=8.6)
        if i:
            _arrow(ax, xs[i] - gap + 0.04, y2 + h / 2, xs[i] - 0.02, y2 + h / 2)

    # حلقة التغذية الراجعة: من التقييم إلى التدريب (أسفل الصف الثاني)
    y_fb = 1.02
    ax.plot([xs[2] + w / 2, xs[2] + w / 2, xs[1] + w / 2],
            [y2, y_fb, y_fb], color=RED, lw=1.3, ls="--", zorder=1)
    _arrow(ax, xs[1] + w / 2, y_fb, xs[1] + w / 2, y2 - 0.02, color=RED)
    ax.text(xs[1] + w / 2 + 0.2, y_fb + 0.12, "tune hyper-parameters",
            ha="left", fontsize=8, color=RED, style="italic")

    ax.text(5.8, 4.82, "End-to-end pipeline for corn leaf disease detection",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5.8, 0.42, "Outputs: trained .keras model  ·  metrics.json  ·  report figures  ·  prediction CLI",
            ha="center", fontsize=8.8, color="#566573")

    fig.tight_layout()
    out = out or FIGURES_DIR / "fig00_pipeline.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[شكل] {out.name}")
    return out


def architecture_diagram(out: Path | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(11.6, 3.9))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 3.9)
    ax.axis("off")

    layers = [
        ("Input\n224×224×3", "#5D6D7E", "224²×3"),
        ("Augmentation\n(online)", ORANGE, "224²×3"),
        ("Backbone\nMobileNetV2", BLUE, "7²×1280"),
        ("Global Avg\nPooling", "#8E44AD", "1280"),
        ("BatchNorm\n+ Dropout", "#8E44AD", "1280"),
        ("Dense 4\nsoftmax", GREEN, "4"),
        ("Prediction\n4 classes", RED, "4"),
    ]
    n, w, h, y, x0 = len(layers), 1.42, 1.12, 1.55, 0.28
    gap = (11.6 - 2 * x0 - n * w) / (n - 1)

    for i, (label, colour, dim) in enumerate(layers):
        x = x0 + i * (w + gap)
        _box(ax, x, y, w, h, label, colour, fontsize=7.6)
        ax.text(x + w / 2, y - 0.24, dim, ha="center", fontsize=7,
                color="#566573", style="italic")
        if i:
            _arrow(ax, x - gap + 0.04, y + h / 2, x - 0.02, y + h / 2)

    ax.text(5.8, 3.5, "Model architecture (transfer-learning configuration)",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5.8, 0.86,
            "Frozen ImageNet backbone → fine-tune top 40 layers  ·  AdamW  ·  label smoothing 0.05",
            ha="center", fontsize=8.4, color="#566573", style="italic")

    fig.tight_layout()
    out = out or FIGURES_DIR / "fig17_architecture.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[شكل] {out.name}")
    return out


if __name__ == "__main__":
    pipeline_diagram()
    architecture_diagram()
