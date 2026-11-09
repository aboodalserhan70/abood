"""
تدريب نموذج تصنيف أمراض أوراق الذرة.

مثال التشغيل:
    python -m src.train --epochs 40 --backbone mobilenetv2 --img-size 224
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import CLASS_NAMES, FIGURES_DIR, MODELS_DIR, OUTPUT_DIR, TrainConfig, ensure_dirs
from .data import compute_class_weights, make_dataset, scan_dataset, split_dataset
from .model import build_model, compile_model, count_params, unfreeze_top


# ---------------------------------------------------------------------------
def plot_history(history: dict, out: Path) -> Path:
    """يرسم منحنيات الخسارة والدقة لمراحل التدريب."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    epochs = range(1, len(history["loss"]) + 1)

    axes[0].plot(epochs, history["loss"], label="train", color="#2E86C1", lw=1.8)
    axes[0].plot(epochs, history["val_loss"], label="validation", color="#C0392B", lw=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curve", fontweight="bold")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, history["accuracy"], label="train", color="#2E86C1", lw=1.8)
    axes[1].plot(epochs, history["val_accuracy"], label="validation", color="#C0392B", lw=1.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy curve", fontweight="bold")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[شكل] {out.name}")
    return out


def plot_learning_rate(history: dict, out: Path) -> Path | None:
    if "learning_rate" not in history:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.plot(range(1, len(history["learning_rate"]) + 1), history["learning_rate"],
            color="#8E44AD", lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate (log scale)")
    ax.set_title("Learning-rate schedule", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def merge_histories(*histories: dict) -> dict:
    merged: dict[str, list] = {}
    for hist in histories:
        for key, values in hist.items():
            merged.setdefault(key, []).extend(list(values))
    return merged


# ---------------------------------------------------------------------------
def train(cfg: TrainConfig, verbose: int = 1) -> dict:
    ensure_dirs()
    tf.keras.utils.set_random_seed(cfg.seed)

    df = scan_dataset(cfg.data_dir)
    split = split_dataset(df, cfg.val_split, cfg.test_split, cfg.seed)
    print(split.summary().to_string())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    split.train[["path", "label"]].to_csv(OUTPUT_DIR / "train_split.csv", index=False)
    split.val[["path", "label"]].to_csv(OUTPUT_DIR / "val_split.csv", index=False)
    split.test[["path", "label"]].to_csv(OUTPUT_DIR / "test_split.csv", index=False)

    train_ds = make_dataset(split.train, cfg.img_size, cfg.batch_size, training=True,
                            seed=cfg.seed, num_parallel_calls=cfg.num_parallel_calls)
    val_ds = make_dataset(split.val, cfg.img_size, cfg.batch_size,
                          seed=cfg.seed, num_parallel_calls=cfg.num_parallel_calls)

    model = build_model(len(CLASS_NAMES), cfg.img_size, cfg.backbone, cfg.dropout)
    compile_model(model, cfg.learning_rate, cfg.label_smoothing)
    params = count_params(model)
    if verbose:
        model.summary()

    class_weight = compute_class_weights(split.train) if cfg.use_class_weights else None
    if class_weight:
        print(f"[معلومة] أوزان الفئات: { {CLASS_NAMES[k]: round(v, 2) for k, v in class_weight.items()} }")

    ckpt_path = MODELS_DIR / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=max(6, cfg.epochs // 4),
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=4, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(OUTPUT_DIR / "logs" / "training_log.csv"),
    ]

    t0 = time.time()
    hist1 = model.fit(
        train_ds, validation_data=val_ds, epochs=cfg.epochs,
        class_weight=class_weight, callbacks=callbacks, verbose=verbose,
    ).history
    stage1_seconds = time.time() - t0

    hist_all = hist1
    stage2_seconds = 0.0
    if cfg.backbone != "scratch" and cfg.fine_tune_epochs > 0:
        print("\n=== مرحلة الضبط الدقيق (Fine-tuning) ===")
        unfreeze_top(model, n_layers=40)
        compile_model(model, cfg.fine_tune_lr, cfg.label_smoothing)
        t1 = time.time()
        hist2 = model.fit(
            train_ds, validation_data=val_ds, epochs=cfg.fine_tune_epochs,
            class_weight=class_weight, callbacks=callbacks, verbose=verbose,
        ).history
        stage2_seconds = time.time() - t1
        hist_all = merge_histories(hist1, hist2)

    model.save(MODELS_DIR / "final_model.keras")
    (MODELS_DIR / "class_names.json").write_text(
        json.dumps(CLASS_NAMES, ensure_ascii=False, indent=2), encoding="utf-8")

    history_path = OUTPUT_DIR / "history.json"
    history_path.write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in hist_all.items()}, indent=2),
        encoding="utf-8")

    plot_history(hist_all, FIGURES_DIR / "fig07_training_curves.png")
    plot_learning_rate(hist_all, FIGURES_DIR / "fig08_learning_rate.png")

    best_val = float(max(hist_all.get("val_accuracy", [0.0])))
    report = {
        "backbone": cfg.backbone,
        "img_size": cfg.img_size,
        "batch_size": cfg.batch_size,
        "epochs_stage1": len(hist1.get("loss", [])),
        "epochs_stage2": len(hist_all.get("loss", [])) - len(hist1.get("loss", [])),
        "best_val_accuracy": best_val,
        "final_train_accuracy": float(hist_all["accuracy"][-1]),
        "final_val_loss": float(hist_all["val_loss"][-1]),
        "parameters": params,
        "class_weight": ({CLASS_NAMES[k]: round(v, 4) for k, v in class_weight.items()}
                         if class_weight else {}),
        "train_seconds": round(stage1_seconds + stage2_seconds, 1),
        "train_seconds_stage1": round(stage1_seconds, 1),
        "train_seconds_stage2": round(stage2_seconds, 1),
        "n_train": int(len(split.train)),
        "n_val": int(len(split.val)),
        "n_test": int(len(split.test)),
        "tensorflow": tf.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (OUTPUT_DIR / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[تم] أفضل دقة تحقّق: {best_val:.4f} | "
          f"زمن التدريب: {report['train_seconds']}s")
    return report


# ---------------------------------------------------------------------------
def parse_args() -> tuple[TrainConfig, int]:
    p = argparse.ArgumentParser(description="تدريب نموذج كشف أمراض أوراق الذرة")
    p.add_argument("--data", type=Path, default=TrainConfig.data_dir)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--fine-tune-epochs", type=int, default=TrainConfig.fine_tune_epochs)
    p.add_argument("--img-size", type=int, default=TrainConfig.img_size)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--backbone", default=TrainConfig.backbone,
                   choices=["scratch", "mobilenetv2", "efficientnetb0"])
    p.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--verbose", type=int, default=1)
    a = p.parse_args()

    cfg = TrainConfig(
        data_dir=a.data, epochs=a.epochs, fine_tune_epochs=a.fine_tune_epochs,
        img_size=a.img_size, batch_size=a.batch_size, backbone=a.backbone,
        learning_rate=a.lr, seed=a.seed,
    )
    return cfg, a.verbose


if __name__ == "__main__":
    cfg, verbose = parse_args()
    train(cfg, verbose=verbose)
