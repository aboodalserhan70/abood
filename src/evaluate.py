"""
تقييم النموذج على مجموعة الاختبار: مصفوفة الالتباس، الدقة/الاستدعاء/F1،
معرض التوقعات، وخرائط Grad-CAM.
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
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from .config import CLASS_INFO, CLASS_NAMES, FIGURES_DIR, MODELS_DIR, OUTPUT_DIR
from .data import make_dataset, scan_dataset, split_dataset
from .gradcam import compute_gradcam, overlay_heatmap
from . import model as _model_module  # noqa: F401  (تسجيل الدوال المخصّصة عند التحميل)

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 118, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "grid.alpha": 0.25})


# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, out: Path, normalize: bool = False) -> Path:
    matrix = cm.astype(float)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1e-9))
    labels = [CLASS_INFO[c]["en"].replace(" ", "\n") for c in CLASS_NAMES]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""),
                 fontweight="bold")
    ax.grid(False)
    threshold = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f"{value:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if value > threshold else "#1B2631",
                    fontweight="bold" if i == j else "normal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[شكل] {out.name}")
    return out


def plot_per_class_metrics(metrics: dict, out: Path) -> Path:
    classes = CLASS_NAMES
    x = np.arange(len(classes))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for i, (key, label, color) in enumerate(
            [("precision", "Precision", "#2E86C1"),
             ("recall", "Recall", "#27AE60"),
             ("f1", "F1-score", "#C0392B")]):
        vals = [metrics[c][key] for c in classes]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color,
                      edgecolor="black", linewidth=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:.2f}",
                    ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_INFO[c]["en"] for c in classes], rotation=12, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision / recall / F1", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[شكل] {out.name}")
    return out


def plot_confidence(df_test: pd.DataFrame, probs: np.ndarray, out: Path) -> Path:
    """توزيع الثقة: يصحّح مقابل خطأ — يكشف فرط الثقة."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == df_test["label_id"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    bins = np.linspace(0, 1, 11)
    axes[0].hist(conf[correct], bins=bins, alpha=0.8, label="correct",
                 color="#27AE60", edgecolor="black", linewidth=0.4)
    axes[0].hist(conf[~correct], bins=bins, alpha=0.8, label="incorrect",
                 color="#C0392B", edgecolor="black", linewidth=0.4)
    axes[0].set_xlabel("Prediction confidence")
    axes[0].set_ylabel("Number of images")
    axes[0].set_title("Confidence distribution", fontweight="bold")
    axes[0].legend(frameon=False)

    order = np.argsort(-conf)
    axes[1].bar(range(len(order)), conf[order],
                color=["#27AE60" if correct[i] else "#C0392B" for i in order],
                edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel("Test image (sorted by confidence)")
    axes[1].set_ylabel("Confidence")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Per-image confidence (green = correct)", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[شكل] {out.name}")
    return out


def plot_prediction_gallery(df_test: pd.DataFrame, probs: np.ndarray, out: Path,
                            max_items: int = 8) -> Path:
    n = min(len(df_test), max_items)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.9 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue
        row = df_test.iloc[i]
        pred = int(probs[i].argmax())
        conf = float(probs[i].max())
        true_id = int(row["label_id"])
        with Image.open(row["path"]) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((420, 420), Image.LANCZOS)   # تصغير لتقليل حجم الشكل
            ax.imshow(thumb)
        color = "#1E8449" if pred == true_id else "#C0392B"
        ax.set_title(f"true: {CLASS_INFO[CLASS_NAMES[true_id]]['en']}\n"
                     f"pred: {CLASS_INFO[CLASS_NAMES[pred]]['en']} ({conf:.0%})",
                     fontsize=7.5, color=color, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2.2)
    fig.suptitle("Test-set predictions", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[شكل] {out.name}")
    return out


def plot_gradcam_gallery(model, df_test: pd.DataFrame, probs: np.ndarray,
                         img_size: int, out: Path, max_items: int = 4) -> Path | None:
    """يعرض الصورة الأصلية بجانب خريطة Grad-CAM لأول صور الاختبار."""
    n = min(len(df_test), max_items)
    if n == 0:
        return None
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.2))
    axes = np.atleast_2d(axes)

    for i in range(n):
        row = df_test.iloc[i]
        with Image.open(row["path"]) as im:
            arr = np.asarray(im.convert("RGB").resize((img_size, img_size)),
                             dtype=np.float32) / 255.0
        pred = int(probs[i].argmax())
        try:
            heat = compute_gradcam(model, arr[np.newaxis, ...], class_index=pred)
            overlay = overlay_heatmap(arr, heat)
        except Exception as exc:  # noqa: BLE001
            print(f"[تحذير] تعذّر حساب Grad-CAM: {exc}")
            overlay = arr
        axes[0, i].imshow(arr)
        axes[0, i].set_title(f"input #{i + 1}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"Grad-CAM: {CLASS_INFO[CLASS_NAMES[pred]]['en']}",
                             fontsize=8, fontweight="bold")
        axes[1, i].axis("off")

    fig.suptitle("Grad-CAM: where the model looks", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[شكل] {out.name}")
    return out


# ---------------------------------------------------------------------------
def evaluate(model_path: Path | None = None, data_dir: Path | None = None,
             img_size: int = 224, batch_size: int = 16, seed: int = 42,
             make_gradcam: bool = True) -> dict:
    from .config import DATA_DIR, TrainConfig

    model_path = model_path or MODELS_DIR / "best_model.keras"
    if not model_path.exists():
        model_path = MODELS_DIR / "final_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            "لم يتم العثور على نموذج مدرَّب. شغّل `python -m src.train` أولاً.")

    print(f"[معلومة] تحميل النموذج: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    df = scan_dataset(data_dir or DATA_DIR)
    split = split_dataset(df, TrainConfig.val_split, TrainConfig.test_split, seed)
    df_test = split.test.reset_index(drop=True)

    test_ds = make_dataset(df_test, img_size, batch_size, seed=seed)
    probs = model.predict(test_ds, verbose=0)
    y_true = df_test["label_id"].to_numpy()
    y_pred = probs.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=[CLASS_INFO[c]["en"] for c in CLASS_NAMES],
        output_dict=True, zero_division=0)

    per_class = {
        cls: {
            "precision": float(report[CLASS_INFO[cls]["en"]]["precision"]),
            "recall": float(report[CLASS_INFO[cls]["en"]]["recall"]),
            "f1": float(report[CLASS_INFO[cls]["en"]]["f1-score"]),
            "support": int(report[CLASS_INFO[cls]["en"]]["support"]),
        }
        for cls in CLASS_NAMES
    }

    metrics = {
        "model": str(model_path.name),
        "n_test": int(len(df_test)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "predictions": [
            {
                "file": Path(df_test.iloc[i]["path"]).name,
                "true": CLASS_NAMES[int(y_true[i])],
                "predicted": CLASS_NAMES[int(y_pred[i])],
                "confidence": round(float(probs[i].max()), 4),
                "probabilities": {CLASS_NAMES[j]: round(float(p), 4)
                                  for j, p in enumerate(probs[i])},
            }
            for i in range(len(df_test))
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        [{"file": p["file"], "true": p["true"], "predicted": p["predicted"],
          "confidence": p["confidence"]} for p in metrics["predictions"]]
    ).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    plot_confusion_matrix(cm, FIGURES_DIR / "fig09_confusion_matrix.png")
    plot_confusion_matrix(cm, FIGURES_DIR / "fig10_confusion_matrix_norm.png", normalize=True)
    plot_per_class_metrics(per_class, FIGURES_DIR / "fig11_per_class_metrics.png")
    plot_confidence(df_test, probs, FIGURES_DIR / "fig12_confidence.png")
    plot_prediction_gallery(df_test, probs, FIGURES_DIR / "fig13_prediction_gallery.png")
    if make_gradcam:
        plot_gradcam_gallery(model, df_test, probs, img_size,
                             FIGURES_DIR / "fig14_gradcam.png")

    print("\n===== ملخص التقييم =====")
    print(f"الدقة الكلية (accuracy): {metrics['accuracy']:.4f}")
    print(f"متوسط F1 (macro)      : {metrics['macro_f1']:.4f}")
    print(f"كابا كوهين            : {metrics['cohen_kappa']:.4f}")
    print(f"عدد صور الاختبار      : {metrics['n_test']}")
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="تقييم نموذج كشف أمراض أوراق الذرة")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gradcam", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    evaluate(a.model, a.data, a.img_size, a.batch_size, a.seed, not a.no_gradcam)
