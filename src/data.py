"""
تحميل مجموعة البيانات وتقسيمها وبناء خطوط أنابيب tf.data.
Dataset discovery, stratified splitting and tf.data pipelines.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import CLASS_NAMES

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# 1) فحص المجلد واستخراج مسارات الصور
# ---------------------------------------------------------------------------
def scan_dataset(data_dir: Path | str) -> pd.DataFrame:
    """يعيد جدولاً بمسارات الصور وفئاتها. يتجاهل أي مجلد غير معروف."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"لم يتم العثور على مجلد البيانات: {data_dir}")

    rows: list[dict] = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = class_dir.name
        if label not in CLASS_NAMES:
            print(f"[تحذير] تم تجاهل مجلد غير معروف: {label}")
            continue
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.suffix.lower() in IMG_EXTENSIONS and img_path.is_file():
                rows.append({"path": str(img_path), "label": label})

    if not rows:
        raise RuntimeError("لم يتم العثور على أي صورة صالحة داخل مجلد البيانات.")

    df = pd.DataFrame(rows)
    df["label_id"] = df["label"].map({name: i for i, name in enumerate(CLASS_NAMES)})
    return df


# ---------------------------------------------------------------------------
# 2) تقسيم طبقي (train / val / test)
# ---------------------------------------------------------------------------
@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        def counts(frame: pd.DataFrame) -> dict[str, int]:
            c = frame["label"].value_counts()
            return {name: int(c.get(name, 0)) for name in CLASS_NAMES}

        table = pd.DataFrame(
            {
                "train": counts(self.train),
                "val": counts(self.val),
                "test": counts(self.test),
            }
        )
        table["total"] = table.sum(axis=1)
        return table


def _split_one_class(df: pd.DataFrame, val_split: float, test_split: float,
                     rng: random.Random) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = list(range(len(df)))
    rng.shuffle(idx)
    n = len(idx)
    n_test = int(round(n * test_split))
    n_val = int(round(n * val_split))

    # نضمن وجود صورة واحدة على الأقل في كل مجموعة متى كان ذلك ممكناً
    if n >= 3:
        n_test = max(1, n_test)
        n_val = max(1, n_val)
    while n_val + n_test >= n and (n_val > 1 or n_test > 1):
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


def split_dataset(df: pd.DataFrame, val_split: float = 0.15,
                  test_split: float = 0.15, seed: int = 42) -> Split:
    """تقسيم طبقي يحافظ على نسب الفئات في كل مجموعة."""
    rng = random.Random(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, group in df.groupby("label", sort=False):
        tr, va, te = _split_one_class(group.reset_index(drop=True), val_split, test_split, rng)
        parts["train"].append(tr)
        parts["val"].append(va)
        parts["test"].append(te)

    split = Split(
        train=pd.concat(parts["train"]).sample(frac=1, random_state=seed).reset_index(drop=True),
        val=pd.concat(parts["val"]).sample(frac=1, random_state=seed).reset_index(drop=True),
        test=pd.concat(parts["test"]).sample(frac=1, random_state=seed).reset_index(drop=True),
    )
    return split


# ---------------------------------------------------------------------------
# 3) قراءة الصور وتوسيع البيانات (Augmentation)
# ---------------------------------------------------------------------------
def decode_image(path: tf.Tensor, label: tf.Tensor, img_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)          # [0, 1]
    image = tf.image.resize(image, (img_size, img_size), antialias=True)
    image = tf.ensure_shape(image, (img_size, img_size, 3))
    return image, label


def build_augmentation(img_size: int, seed: int | None = None) -> tf.keras.Sequential:
    """
    تحسينات مناسبة لصور الأوراق: تدوير بسيط، انعكاس أفقي، تقريب/تبعيد،
    تغيّر طفيف في الإضاءة والتباين. تم تجنّب القص العشوائي القوي لأنه يحذف
    الآفة نفسها من الصورة.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.06, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomZoom(0.12, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomTranslation(0.06, 0.06, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomContrast(0.15, seed=seed),
            tf.keras.layers.RandomBrightness(
                (-0.10, 0.10), value_range=(0.0, 1.0), seed=seed
            ),
        ],
        name="augmentation",
    )


def make_dataset(df: pd.DataFrame, img_size: int, batch_size: int,
                 training: bool = False, seed: int = 42,
                 num_parallel_calls: int = 2) -> tf.data.Dataset:
    """يبني tf.data.Dataset من جدول المسارات."""
    paths = df["path"].to_numpy()
    labels = df["label_id"].to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(lambda p, y: decode_image(p, y, img_size),
                num_parallel_calls=num_parallel_calls)

    if training:
        augment = build_augmentation(img_size, seed=seed)
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=num_parallel_calls)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def compute_class_weights(df: pd.DataFrame) -> dict[int, float]:
    """أوزان معاكسة للتكرار لمعالجة عدم توازن الفئات."""
    counts = df["label_id"].value_counts().to_dict()
    total = sum(counts.values())
    n_classes = len(CLASS_NAMES)
    return {cid: total / (n_classes * max(c, 1)) for cid, c in counts.items()}
