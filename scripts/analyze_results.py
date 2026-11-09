"""
تحليل إضافي لمخرجات التشغيل الحقيقية (validation run).

يقرأ الملفات الموجودة في outputs/validation_run/ وينتج:
  • منحنيات التدريب للمرحلتين
  • رسم مقاييس كل فئة
  • تحليل الثقة وقابلية الفرز (triage) — مع فترات ثقة ويلسون
  • ملخص إحصائي في JSON

الاستخدام:
    python scripts/analyze_results.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
RUN = ROOT / "outputs" / "validation_run"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 125, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
})

CLASS_ORDER = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
CLASS_COLOR = {
    "Blight": "#C0392B", "Common_Rust": "#D68910",
    "Gray_Leaf_Spot": "#7F8C8D", "Healthy": "#27AE60",
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """فترة ثقة ويلسون لنسبة (أدق من الفترة الطبيعية للمجاميع الصغيرة)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[شكل] {path.name}")
    return path


# ---------------------------------------------------------------------------
def plot_curves(history: dict, out: Path) -> Path:
    """منحنيات الدقة والخسارة للمرحلتين مع تمييز المرحلة المختارة."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))

    stages = [("stage1", "Stage 1 (frozen backbone)", "#1E6F5C", "-", "o"),
              ("stage2", "Stage 2 (fine-tuning)", "#C0392B", "--", "s")]

    offset = 0
    for key, label, colour, ls, marker in stages:
        h = history.get(key)
        if not h:
            continue
        epochs = np.arange(1, len(h["loss"]) + 1) + offset
        axes[0].plot(epochs, h["loss"], ls=ls, marker=marker, ms=4, lw=1.7,
                     color=colour, label=f"{label} — train")
        axes[0].plot(epochs, h["val_loss"], ls=ls, marker=marker, ms=4, lw=1.3,
                     color=colour, alpha=0.55, label=f"{label} — val")
        axes[1].plot(epochs, np.array(h["accuracy"]) * 100, ls=ls, marker=marker,
                     ms=4, lw=1.7, color=colour, label=f"{label} — train")
        axes[1].plot(epochs, np.array(h["val_accuracy"]) * 100, ls=ls, marker=marker,
                     ms=4, lw=1.3, color=colour, alpha=0.55, label=f"{label} — val")
        offset = epochs[-1]

    if "stage2" in history and history["stage2"]:
        split_at = len(history["stage1"]["loss"]) + 0.5
        for ax in axes:
            ax.axvline(split_at, color="#566573", lw=1, ls=":")
        axes[0].text(split_at + 0.15, axes[0].get_ylim()[0], " stage 2 →",
                     fontsize=7.5, color="#566573", va="bottom")

    axes[1].axhline(93.787, color="#1E6F5C", lw=1, ls=":", alpha=0.8)
    axes[1].text(0.02, 0.885, "dotted line = best stage-1 validation accuracy (93.79%)",
                 transform=axes[1].transAxes, ha="left", fontsize=7.2,
                 color="#1E6F5C")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Categorical cross-entropy loss")
    axes[0].set_title("Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7, ncol=1, loc="upper right")
    axes[1].legend(frameon=False, fontsize=7, ncol=1, loc="lower right")
    axes[1].set_ylim(78, 96.4)
    fig.suptitle("Training dynamics — two-stage transfer learning",
                 fontweight="bold", y=1.02)
    return _save(fig, out)


def plot_per_class(report: pd.DataFrame, out: Path) -> Path:
    """مقاييس كل فئة مع فترات ثقة ويلسون للاستدعاء."""
    rows = []
    for cls in CLASS_ORDER:
        r = report.loc[cls]
        n = int(r["support"])
        recall = float(r["recall"])
        k = int(round(recall * n))
        lo, hi = wilson_ci(k, n)
        rows.append({"class": cls, "precision": float(r["precision"]),
                     "recall": recall, "f1": float(r["f1-score"]),
                     "support": n, "ci_lo": lo, "ci_hi": hi})

    x = np.arange(len(rows))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    for i, (key, label, colour) in enumerate(
            [("precision", "Precision", "#2E86C1"),
             ("recall", "Recall", "#1E6F5C"),
             ("f1", "F1-score", "#C0392B")]):
        vals = [r[key] for r in rows]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label,
                      color=colour, edgecolor="black", linewidth=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", fontsize=6.8)

    # فترة الثقة على الاستدعاء
    ax.errorbar([r["precision"] for r in rows] and x, [r["recall"] for r in rows],
                yerr=[[r["recall"] - r["ci_lo"] for r in rows],
                      [r["ci_hi"] - r["recall"] for r in rows]],
                fmt="none", ecolor="#1B2631", elinewidth=1.1, capsize=3, zorder=5)

    labels = [f"{r['class']}\n(n={r['support']})" for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision / recall / F1 on the validation set\n"
                 "(error bars = 95% Wilson CI on recall)", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    return _save(fig, out)


def plot_confidence(preds: pd.DataFrame, out: Path) -> Path:
    """توزيع الثقة + الدقة حسب شريحة الثقة + منحنى الفرز."""
    correct = preds["correct"].to_numpy()
    conf = preds["max_softmax_score"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))

    # (أ) توزيع الثقة
    bins = np.linspace(0, 1, 21)
    axes[0].hist(conf[correct], bins=bins, color="#27AE60", alpha=0.85,
                 edgecolor="black", linewidth=0.3, label=f"correct (n={correct.sum()})")
    axes[0].hist(conf[~correct], bins=bins, color="#C0392B", alpha=0.85,
                 edgecolor="black", linewidth=0.3, label=f"incorrect (n={(~correct).sum()})")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Max. softmax score")
    axes[0].set_ylabel("Number of images (log scale)")
    axes[0].set_title("Confidence distribution", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7.5)

    # (ب) الدقة حسب شريحة الثقة
    edges = [0, .5, .6, .7, .8, .9, 1.001]
    labels, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        labels.append(f"{lo:.1f}–{min(hi, 1.0):.1f}")
        accs.append(correct[mask].mean() * 100)
        counts.append(int(mask.sum()))
    bars = axes[1].bar(labels, accs, color="#2E86C1", edgecolor="black", linewidth=0.3)
    for b, a, n in zip(bars, accs, counts):
        axes[1].text(b.get_x() + b.get_width() / 2, a + 1.5, f"{a:.0f}%\nn={n}",
                     ha="center", fontsize=7)
    axes[1].axhline(correct.mean() * 100, color="#C0392B", ls=":", lw=1.2)
    axes[1].text(0.02, correct.mean() * 100 + 1.2, "overall accuracy",
                 color="#C0392B", fontsize=7.2)
    axes[1].set_ylim(0, 112)
    axes[1].set_xlabel("Confidence bin")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy vs. confidence", fontweight="bold")

    # (ج) منحنى المقايضة: نسبة الصور المُحالة للمراجعة مقابل نسبة الأخطاء المكتشفة
    thresholds = np.linspace(0.3, 0.999, 140)
    reviewed, caught = [], []
    total_err = (~correct).sum()
    for t in thresholds:
        below = conf < t
        reviewed.append(below.mean() * 100)
        caught.append((below & ~correct).sum() / total_err * 100)
    axes[2].plot(reviewed, caught, color="#8E44AD", lw=2)
    for t, mark in [(0.7, "0.70"), (0.9, "0.90")]:
        below = conf < t
        axes[2].scatter([below.mean() * 100],
                        [(below & ~correct).sum() / total_err * 100],
                        s=45, color="#C0392B", zorder=5)
        axes[2].annotate(f"thr {mark}\n{below.mean() * 100:.0f}% reviewed\n"
                         f"{(below & ~correct).sum() / total_err * 100:.0f}% of errors caught",
                         (below.mean() * 100, (below & ~correct).sum() / total_err * 100),
                         textcoords="offset points", xytext=(12, -6), fontsize=7)
    axes[2].set_xlabel("Share of images sent for human review (%)")
    axes[2].set_ylabel("Share of errors caught (%)")
    axes[2].set_title("Reject-option / triage curve", fontweight="bold")
    axes[2].set_xlim(0, 40)
    axes[2].set_ylim(0, 102)

    fig.suptitle("Confidence analysis on the validation set (n = 837)",
                 fontweight="bold", y=1.03)
    return _save(fig, out)


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="تحليل مخرجات التشغيل الحقيقية")
    p.add_argument("--run-dir", type=Path, default=RUN)
    p.add_argument("--figures-dir", type=Path, default=FIG)
    a = p.parse_args()

    run, figures = a.run_dir, a.figures_dir
    figures.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(run / "validation_predictions.csv")
    preds["correct"] = preds["class_name"] == preds["predicted_class"]
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    report = pd.read_csv(run / "classification_report.csv", index_col=0)
    metrics = json.loads((run / "report_metrics.json").read_text(encoding="utf-8"))

    plot_curves(history, figures / "results_curves.png")
    plot_per_class(report, figures / "results_per_class.png")
    plot_confidence(preds, figures / "results_confidence.png")

    # ملخص إحصائي موسّع
    n = len(preds)
    k = int(preds["correct"].sum())
    acc_lo, acc_hi = wilson_ci(k, n)
    per_class = {}
    for cls in CLASS_ORDER:
        sub = preds[preds["class_name"] == cls]
        if len(sub) == 0:
            continue
        kk, nn = int(sub["correct"].sum()), len(sub)
        lo, hi = wilson_ci(kk, nn)
        per_class[cls] = {
            "support": nn, "correct": kk, "recall": kk / nn,
            "recall_ci95": [round(lo, 4), round(hi, 4)],
            "mean_confidence": round(float(sub["max_softmax_score"].mean()), 4),
        }

    errors = preds[~preds["correct"]]
    summary = {
        "n_validation": n,
        "n_correct": k,
        "accuracy": round(k / n, 4),
        "accuracy_ci95": [round(acc_lo, 4), round(acc_hi, 4)],
        "mean_confidence": round(float(preds["max_softmax_score"].mean()), 4),
        "mean_confidence_correct": round(float(preds.loc[preds["correct"], "max_softmax_score"].mean()), 4),
        "mean_confidence_incorrect": round(float(errors["max_softmax_score"].mean()), 4),
        "n_errors": int(len(errors)),
        "n_confident_errors_gt_0_9": int(((errors["max_softmax_score"] > 0.9)).sum()),
        "per_class": per_class,
        "error_pairs": {
            f"{a_} -> {b_}": int(v)
            for (a_, b_), v in errors.groupby(["class_name", "predicted_class"]).size().items()
        },
        "triage": {},
        "dominant_error_share": round(
            float(errors.groupby(["class_name", "predicted_class"]).size().max() / len(errors)), 4),
        "reported_metrics": {
            "accuracy_percent": metrics["validation_accuracy_percent"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "selected_stage": metrics["selected_stage"],
        },
    }
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        below = preds["max_softmax_score"] < t
        summary["triage"][f"{t:.1f}"] = {
            "reviewed_percent": round(float(below.mean() * 100), 2),
            "errors_caught_percent": round(float((below & ~preds["correct"]).sum() / len(errors) * 100), 2),
            "accuracy_if_accepted": round(float(preds.loc[~below, "correct"].mean() * 100), 2),
        }

    out = run / "analysis_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ملف] {out.relative_to(ROOT)}")

    print("\n=== ملخص ===")
    print(f"دقة التحقق: {summary['accuracy'] * 100:.2f}% "
          f"(فترة ثقة 95%: {acc_lo * 100:.1f}–{acc_hi * 100:.1f}%)")
    print(f"متوسط الثقة — صحيح {summary['mean_confidence_correct']:.3f} | "
          f"خطأ {summary['mean_confidence_incorrect']:.3f}")
    print(f"أخطاء واثقة (>0.9): {summary['n_confident_errors_gt_0_9']} من {summary['n_errors']}")
    for cls, s in per_class.items():
        print(f"  {cls:15s} recall {s['recall'] * 100:5.1f}% "
              f"[{s['recall_ci95'][0] * 100:.0f}–{s['recall_ci95'][1] * 100:.0f}]  n={s['support']}")
    for t, s in summary["triage"].items():
        print(f"  عتبة {t}: مراجعة {s['reviewed_percent']:.1f}% من الصور تكشف "
              f"{s['errors_caught_percent']:.1f}% من الأخطاء")


if __name__ == "__main__":
    main()
