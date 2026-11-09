"""
توليد مخطط خط الأنابيب المستخدم في التقرير.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import FIGURES_DIR

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 9})

BLUE, GREEN, ORANGE, GRAY, RED = "#2E86C1", "#1E6F5C", "#D68910", "#7F8C8D", "#C0392B"


def _box(ax, x, y, w, h, text, color, text_color="white", fontsize=8.5, radius=0.02):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=1.1, edgecolor="#1B2631", facecolor=color, alpha=0.95, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight="bold", zorder=4,
            linespacing=1.4)


def _arrow(ax, x1, y1, x2, y2, color="#34495E", linestyle="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
        linewidth=1.4, color=color, zorder=2, linestyle=linestyle,
        shrinkA=0.5, shrinkB=0.5))


def pipeline_diagram(out: Path | None = None) -> Path:
    """مخطط خط الأنابيب: صفّان مع موصّلات بزاوية قائمة."""
    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    row1 = [("1  Dataset & cleaning\n4,188 images · SHA-256", BLUE),
            ("2  Pre-processing\n224×224 · rescale [-1, 1]", BLUE),
            ("3  Random split\n80 / 20 · seed 42", GRAY),
            ("4  Augmentation\ntrain only · rotation ±36°", ORANGE)]
    row2 = [("5  MobileNetV2\nImageNet · frozen", GREEN),
            ("6  Two-stage training\nhead → fine-tune", GREEN),
            ("7  Validation\nacc · P/R/F1 · selection", RED),
            ("8  Error analysis\nconfusion · confidence", RED)]

    w, h, gap, x0 = 2.55, 1.05, 0.42, 0.32
    y1, y2 = 3.15, 1.45
    xs = [x0 + i * (w + gap) for i in range(4)]

    for i, (label, colour) in enumerate(row1):
        _box(ax, xs[i], y1, w, h, label, colour, fontsize=8.3)
        if i:
            _arrow(ax, xs[i] - gap + 0.04, y1 + h / 2, xs[i] - 0.02, y1 + h / 2)

    # موصّل بزاوية قائمة: نهاية الصف الأول -> بداية الصف الثاني
    y_mid = (y1 + y2 + h) / 2
    ax.plot([xs[3] + w / 2, xs[3] + w / 2, xs[0] + w / 2],
            [y1, y_mid, y_mid], color="#34495E", lw=1.5, zorder=1)
    _arrow(ax, xs[0] + w / 2, y_mid, xs[0] + w / 2, y2 + h + 0.02)

    for i, (label, colour) in enumerate(row2):
        _box(ax, xs[i], y2, w, h, label, colour, fontsize=8.3)
        if i:
            _arrow(ax, xs[i] - gap + 0.04, y2 + h / 2, xs[i] - 0.02, y2 + h / 2)

    # حلقة التغذية الراجعة: من التقييم إلى التدريب
    y_fb = 1.02
    ax.plot([xs[2] + w / 2, xs[2] + w / 2, xs[1] + w / 2],
            [y2, y_fb, y_fb], color=RED, lw=1.3, ls="--", zorder=1)
    _arrow(ax, xs[1] + w / 2, y_fb, xs[1] + w / 2, y2 - 0.02, color=RED)
    ax.text(xs[1] + w / 2 + 0.2, y_fb + 0.12, "model selection between stages",
            ha="left", fontsize=8, color=RED, style="italic")

    ax.text(5.8, 4.82, "End-to-end pipeline for corn leaf disease detection",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5.8, 0.42, "Outputs: corn.keras (9.24 MiB)  ·  metrics & manifests (CSV/JSON)  ·  report figures",
            ha="center", fontsize=8.6, color="#566573")

    fig.tight_layout()
    out = out or FIGURES_DIR / "fig00_pipeline.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[شكل] {out.name}")
    return out


if __name__ == "__main__":
    pipeline_diagram()
