"""
خط أساس كلاسيكي (Classical baseline): ميزات يدوية + SVM.

الهدف: قياس مدى قابلية الفئات للفصل دون شبكات عصبية، وتوفير مقارنة
مرجعية (baseline) في التقرير. تُستخدم هنا تحقّق تبادلي بإسقاط واحد
(Leave-One-Out) لأن عدد الصور صغير جداً.

مثال التشغيل:
    python -m src.baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import CLASS_INFO, CLASS_NAMES, FIGURES_DIR, OUTPUT_DIR
from .data import scan_dataset

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 118, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

FEATURE_SIZE = 256


# ---------------------------------------------------------------------------
# 1) استخراج الميزات
# ---------------------------------------------------------------------------
def lbp_histogram(gray: np.ndarray, bins: int = 32) -> np.ndarray:
    """نموذج بسيط لـ Local Binary Pattern (8 جيران) مع هيستوغرام مُقلَّص."""
    g = gray.astype(np.float32)
    center = g[1:-1, 1:-1]
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    code = np.zeros_like(center, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        neighbour = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        code |= ((neighbour >= center).astype(np.uint8) << bit)
    hist = np.bincount((code.astype(np.int32).ravel() * bins) // 256, minlength=bins)
    return hist / max(hist.sum(), 1)


def lesion_colour_fractions(rgb: np.ndarray) -> np.ndarray:
    """نسب بكسلات "الآفة" (بني/أصفر/رمادي) مقابل "النسيج السليم" (أخضر)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    greenish = (g > r + 0.04) & (g > b + 0.02)
    yellow = (r > 0.42) & (g > 0.38) & (b < 0.45) & (mx - mn > 0.12) & ~greenish
    brown = (r > g) & (g > b) & (mx < 0.62) & (mx - mn > 0.06)
    grayish = (mx - mn < 0.06) & (mx > 0.22) & (mx < 0.75)
    n = float(rgb.shape[0] * rgb.shape[1])
    return np.array([greenish.sum() / n, yellow.sum() / n,
                     brown.sum() / n, grayish.sum() / n], dtype=np.float32)


def extract_features(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """يعيد (متجه الميزات، الصورة المصغّرة للعرض)."""
    with Image.open(path) as im:
        img = im.convert("RGB").resize((FEATURE_SIZE, FEATURE_SIZE))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    h_hist, _ = np.histogram(hsv[..., 0], bins=12, range=(0, 1))
    s_hist, _ = np.histogram(hsv[..., 1], bins=8, range=(0, 1))
    v_hist, _ = np.histogram(hsv[..., 2], bins=8, range=(0, 1))
    colour_hist = np.concatenate([h_hist, s_hist, v_hist])
    colour_hist = colour_hist / max(colour_hist.sum(), 1)

    ch_stats = np.concatenate([rgb.reshape(-1, 3).mean(axis=0),
                               rgb.reshape(-1, 3).std(axis=0)])

    gy, gx = np.gradient(gray)
    grad_mag = np.hypot(gx, gy)
    lap = (np.roll(gray, 1, 0) + np.roll(gray, -1, 0)
           + np.roll(gray, 1, 1) + np.roll(gray, -1, 1) - 4 * gray)
    gray_hist, _ = np.histogram(gray, bins=16, range=(0, 1))
    p = gray_hist / max(gray_hist.sum(), 1)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    texture = np.array([
        grad_mag.mean(), grad_mag.std(),
        float(lap.var()), float(lap.std()),
        entropy,
        float((grad_mag > 0.12).mean()),          # كثافة الحواف
        float(gray.std()),                        # تباين كلي
    ], dtype=np.float32)

    features = np.concatenate([colour_hist.astype(np.float32), ch_stats,
                               texture, lesion_colour_fractions(rgb),
                               lbp_histogram(gray)])
    return features, rgb


def build_feature_matrix(df) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    rows, thumbs = [], []
    for path in df["path"]:
        feat, thumb = extract_features(path)
        rows.append(feat)
        thumbs.append(thumb)
    return np.asarray(rows, dtype=np.float32), df["label_id"].to_numpy(), thumbs


# ---------------------------------------------------------------------------
# 2) التحقق التبادلي والتقييم
# ---------------------------------------------------------------------------
def run_baseline(data_dir=None) -> dict:
    from .config import DATA_DIR
    df = scan_dataset(data_dir or DATA_DIR).reset_index(drop=True)
    X, y, thumbs = build_feature_matrix(df)
    print(f"[معلومة] مصفوفة الميزات: {X.shape[0]} صورة × {X.shape[1]} ميزة")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=10.0, gamma="scale", kernel="rbf",
                    class_weight="balanced", random_state=42)),
    ])
    loo = LeaveOneOut()
    y_pred = cross_val_predict(pipe, X, y, cv=loo)
    classes_seen = np.unique(y)

    cm = confusion_matrix(y, y_pred, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(
        y, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=[CLASS_INFO[c]["en"] for c in CLASS_NAMES],
        output_dict=True, zero_division=0)

    result = {
        "method": "SVM-RBF على ميزات يدوية (لون HSV + نسيج + LBP)",
        "n_images": int(len(y)),
        "n_features": int(X.shape[1]),
        "cv": "Leave-One-Out",
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(report[CLASS_INFO[CLASS_NAMES[i]]["en"]]["precision"]),
                "recall": float(report[CLASS_INFO[CLASS_NAMES[i]]["en"]]["recall"]),
                "f1": float(report[CLASS_INFO[CLASS_NAMES[i]]["en"]]["f1-score"]),
                "support": int(report[CLASS_INFO[CLASS_NAMES[i]]["en"]]["support"]),
            }
            for i in range(len(CLASS_NAMES)) if i in classes_seen
        },
        "n_correct": int((y == y_pred).sum()),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "baseline_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_feature_space(X, y, thumbs)
    plot_baseline_confusion(cm)

    print(f"\n[خط الأساس] دقة LOO: {result['accuracy']:.4f} "
          f"({result['n_correct']}/{result['n_images']}) | "
          f"F1 الكلي: {result['macro_f1']:.4f}")
    return result


# ---------------------------------------------------------------------------
# 3) الأشكال
# ---------------------------------------------------------------------------
def plot_feature_space(X: np.ndarray, y: np.ndarray, thumbs: list[np.ndarray]) -> Path:
    scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42).fit_transform(scaled)
    perplexity = max(2, min(6, len(X) - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=42, learning_rate="auto").fit_transform(scaled)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, coords, title in [(axes[0], pca, "PCA projection"),
                              (axes[1], tsne, "t-SNE projection")]:
        for idx, cls in enumerate(CLASS_NAMES):
            mask = y == idx
            if not mask.any():
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1], s=95,
                       color=CLASS_INFO[cls]["color"], label=CLASS_INFO[cls]["en"],
                       edgecolor="black", linewidth=0.6, alpha=0.92)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Hand-crafted feature space (colour + texture descriptors)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "fig15_feature_space.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[شكل] {out.name}")
    return out


def plot_baseline_confusion(cm: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    im = ax.imshow(cm, cmap="Greens")
    labels = [CLASS_INFO[c]["en"].replace(" ", "\n") for c in CLASS_NAMES]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Classical baseline — leave-one-out", fontweight="bold")
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() / 2 else "#1B2631",
                    fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    out = FIGURES_DIR / "fig16_baseline_confusion.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[شكل] {out.name}")
    return out


if __name__ == "__main__":
    run_baseline()
