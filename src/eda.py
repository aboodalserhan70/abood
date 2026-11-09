"""
تحليل استكشافي لمجموعة البيانات (EDA) وإنتاج الأشكال العلمية.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import CLASS_INFO, CLASS_NAMES, FIGURES_DIR, ensure_dirs
from .data import scan_dataset

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 118,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
})


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[شكل] {path.relative_to(FIGURES_DIR.parent)}")
    return path


# ---------------------------------------------------------------------------
def plot_class_distribution(df, title_suffix: str = "") -> Path:
    counts = df["label"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int)
    colors = [CLASS_INFO[c]["color"] for c in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bars = ax.bar([CLASS_INFO[c]["en"] for c in CLASS_NAMES], counts.values,
                  color=colors, edgecolor="black", linewidth=0.4, width=0.62)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                str(value), ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Number of images")
    ax.set_title(f"Class distribution{title_suffix}", fontweight="bold")
    ax.set_ylim(0, max(counts.values) * 1.18 if max(counts.values) else 1)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    return _save(fig, "fig01_class_distribution.png")


def plot_sample_grid(df, per_class: int = 5, name: str = "fig02_samples.png") -> Path:
    n_cols = min(per_class, max(df["label"].value_counts().max(), 1))
    fig, axes = plt.subplots(len(CLASS_NAMES), n_cols,
                             figsize=(1.75 * n_cols, 1.9 * len(CLASS_NAMES)))
    axes = np.atleast_2d(axes)
    for r, cls in enumerate(CLASS_NAMES):
        subset = df[df["label"] == cls].head(n_cols)
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(subset):
                img = Image.open(subset.iloc[c]["path"]).convert("RGB")
                img.thumbnail((420, 420), Image.LANCZOS)   # تصغير المصغّرات لتقليل حجم الملف
                ax.imshow(img)
            if c == 0:
                info = CLASS_INFO[cls]
                ax.set_title(info["en"], fontsize=9, fontweight="bold",
                             color=info["color"], loc="left")
    fig.suptitle("Sample images per class", fontweight="bold", y=1.005)
    return _save(fig, name)


def plot_size_analysis(df) -> tuple[Path, dict]:
    sizes, ratios, records = [], [], []
    for _, row in df.iterrows():
        with Image.open(row["path"]) as im:
            w, h = im.size
        sizes.append(w * h)
        ratios.append(w / h)
        records.append({"file": Path(row["path"]).name, "class": row["label"],
                        "width": w, "height": h, "megapixels": round(w * h / 1e6, 2)})
    sizes = np.array(sizes)
    stats = {
        "n_images": int(len(sizes)),
        "min_resolution": [int(min(r["width"] for r in records)),
                           int(min(r["height"] for r in records))],
        "max_resolution": [int(max(r["width"] for r in records)),
                           int(max(r["height"] for r in records))],
        "megapixels": {
            "min": float(sizes.min() / 1e6),
            "max": float(sizes.max() / 1e6),
            "mean": float(sizes.mean() / 1e6),
            "median": float(np.median(sizes) / 1e6),
        },
        "aspect_ratio": {
            "min": float(min(ratios)), "max": float(max(ratios)),
            "mean": float(np.mean(ratios)),
        },
        "files": records,
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for i, cls in enumerate(CLASS_NAMES):
        vals = [r["megapixels"] for r in records if r["class"] == cls]
        if vals:
            axes[0].scatter([i] * len(vals), vals, s=48, alpha=0.85,
                            color=CLASS_INFO[cls]["color"], edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(range(len(CLASS_NAMES)))
    axes[0].set_xticklabels([CLASS_INFO[c]["en"] for c in CLASS_NAMES], rotation=12, ha="right")
    axes[0].set_ylabel("Resolution (megapixels)")
    axes[0].set_title("Image resolution per class", fontweight="bold")

    axes[1].hist(np.array(ratios), bins=min(12, max(4, len(ratios) // 2)),
                 color="#5D6D7E", edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("Aspect ratio (width / height)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Aspect-ratio distribution", fontweight="bold")
    path = _save(fig, "fig03_resolution_analysis.png")
    return path, stats


def plot_color_stats(df) -> Path:
    """متوسط لون الطرفية لكل فئة — مفيد لتفسير انحياز الخلفية."""
    means = {c: [] for c in CLASS_NAMES}
    for _, row in df.iterrows():
        with Image.open(row["path"]) as im:
            arr = np.asarray(im.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
        means[row["label"]].append(arr.reshape(-1, 3).mean(axis=0))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    width = 0.26
    x = np.arange(len(CLASS_NAMES))
    for i, (channel, label) in enumerate(zip(range(3), ["R", "G", "B"])):
        vals = [np.mean([m[i] for m in means[c]]) if means[c] else 0 for c in CLASS_NAMES]
        ax.bar(x + (i - 1) * width, vals, width, label=label,
               color=["#C0392B", "#27AE60", "#2980B9"][i], edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_INFO[c]["en"] for c in CLASS_NAMES], rotation=12, ha="right")
    ax.set_ylabel("Mean channel intensity (0-1)")
    ax.set_title("Mean RGB intensity per class", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    return _save(fig, "fig04_color_statistics.png")


def plot_augmentation_demo(df, img_size: int = 224, name="fig05_augmentation.png") -> Path:
    """عرض مثال على التحسينات المطبقة على صورة واحدة."""
    from .data import build_augmentation
    import tensorflow as tf

    sample = df.sample(1, random_state=7).iloc[0]["path"]
    img = tf.image.resize(
        tf.image.convert_image_dtype(tf.io.decode_image(tf.io.read_file(sample), channels=3), tf.float32),
        (img_size, img_size),
    )
    augment = build_augmentation(img_size, seed=3)

    fig, axes = plt.subplots(1, 6, figsize=(11, 2.3))
    axes[0].imshow(img.numpy())
    axes[0].set_title("Original", fontsize=9, fontweight="bold")
    axes[0].axis("off")
    for i in range(1, 6):
        axes[i].imshow(tf.clip_by_value(augment(tf.expand_dims(img, 0), training=True)[0], 0, 1))
        axes[i].set_title(f"Augmented #{i}", fontsize=9)
        axes[i].axis("off")
    fig.suptitle("Augmentation examples (rotation, flip, zoom, shift, contrast, brightness)",
                 fontweight="bold", y=1.06, fontsize=9.5)
    return _save(fig, name)


def plot_split_summary(split, name="fig06_split_summary.png") -> Path:
    table = split.summary()
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    x = np.arange(len(table))
    bottoms = np.zeros(len(table))
    for part, color in zip(["train", "val", "test"], ["#2E86C1", "#F39C12", "#C0392B"]):
        ax.bar(x, table[part].values, bottom=bottoms, label=part, color=color,
               edgecolor="white", linewidth=0.8)
        bottoms += table[part].values
    for i, total in enumerate(table["total"].values):
        ax.text(i, total + 0.4, str(int(total)), ha="center", fontweight="bold", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=12, ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title("Train / validation / test split per class", fontweight="bold")
    ax.legend(frameon=False, ncol=3)
    return _save(fig, name)


def run_eda(data_dir=None, per_class: int = 5) -> dict:
    from .config import DATA_DIR
    ensure_dirs()
    df = scan_dataset(data_dir or DATA_DIR)
    print(f"[معلومة] عدد الصور: {len(df)} | عدد الفئات: {df['label'].nunique()}")

    plot_class_distribution(df)
    plot_sample_grid(df, per_class=per_class)
    _, size_stats = plot_size_analysis(df)
    plot_color_stats(df)
    plot_augmentation_demo(df)

    class_stats = {
        cls: {
            "count": int((df["label"] == cls).sum()),
            "en": CLASS_INFO[cls]["en"],
            "ar": CLASS_INFO[cls]["ar"],
            "pathogen": CLASS_INFO[cls]["latin"],
        }
        for cls in CLASS_NAMES
    }
    summary = {
        "n_images": int(len(df)),
        "n_classes": int(df["label"].nunique()),
        "classes": class_stats,
        "resolution": size_stats["megapixels"],
        "aspect_ratio": size_stats["aspect_ratio"],
        "files": size_stats["files"],
    }
    out = FIGURES_DIR.parent / "outputs" / "eda_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[معلومة] تم حفظ ملخص EDA في {out}")
    return summary


if __name__ == "__main__":
    run_eda()
