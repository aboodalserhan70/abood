"""
توقّع فئة صورة ورقة ذرة واحدة من سطر الأوامر.

مثال:
    python -m src.predict --image path/to/leaf.jpg
    python -m src.predict --image leaf.jpg --gradcam figures/my_gradcam.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import CLASS_INFO, CLASS_NAMES, MODELS_DIR
from .gradcam import compute_gradcam, overlay_heatmap
from . import model as _model_module  # noqa: F401


def load_model(path: Path | None = None) -> tf.keras.Model:
    path = path or MODELS_DIR / "best_model.keras"
    if not path.exists():
        path = MODELS_DIR / "final_model.keras"
    if not path.exists():
        raise FileNotFoundError("لا يوجد نموذج مدرَّب. شغّل `python -m src.train` أولاً.")
    return tf.keras.models.load_model(path, compile=False)


def preprocess(image_path: str | Path, img_size: int = 224) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    return np.asarray(img, dtype=np.float32)[np.newaxis, ...] / 255.0


def predict_image(image_path: str | Path, model: tf.keras.Model | None = None,
                  img_size: int = 224, top_k: int = 4) -> dict:
    model = model or load_model()
    batch = preprocess(image_path, img_size)
    probs = model.predict(batch, verbose=0)[0]
    order = np.argsort(-probs)[:top_k]

    result = {
        "image": str(image_path),
        "predicted_class": CLASS_NAMES[int(order[0])],
        "predicted_class_ar": CLASS_INFO[CLASS_NAMES[int(order[0])]]["ar"],
        "confidence": float(probs[order[0]]),
        "top_k": [
            {
                "class": CLASS_NAMES[int(i)],
                "label_en": CLASS_INFO[CLASS_NAMES[int(i)]]["en"],
                "label_ar": CLASS_INFO[CLASS_NAMES[int(i)]]["ar"],
                "probability": float(probs[i]),
            }
            for i in order
        ],
    }
    return result


def save_gradcam(image_path: str | Path, out_path: str | Path,
                 model: tf.keras.Model | None = None, img_size: int = 224) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = model or load_model()
    batch = preprocess(image_path, img_size)
    pred = int(model.predict(batch, verbose=0)[0].argmax())
    heat = compute_gradcam(model, batch, class_index=pred)
    overlay = overlay_heatmap(batch[0], heat)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    axes[0].imshow(batch[0])
    axes[0].set_title("Input leaf", fontsize=9)
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM — {CLASS_INFO[CLASS_NAMES[pred]]['en']}",
                      fontsize=9, fontweight="bold")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="توقّع مرض ورقة ذرة من صورة")
    p.add_argument("--image", required=True)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--gradcam", type=Path, default=None,
                   help="مسار لحفظ شكل Grad-CAM")
    p.add_argument("--json", action="store_true", help="إخراج النتيجة بصيغة JSON")
    a = p.parse_args()

    model = load_model(a.model)
    result = predict_image(a.image, model, a.img_size)
    if a.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"\nالصورة: {result['image']}")
        print(f"التوقع: {result['predicted_class_ar']} "
              f"({CLASS_INFO[result['predicted_class']]['en']}) "
              f"— الثقة {result['confidence']:.1%}\n")
        for item in result["top_k"]:
            bar = "█" * int(round(item["probability"] * 30))
            print(f"  {item['label_en']:22s} {item['label_ar']:16s} "
                  f"{item['probability']:6.1%} {bar}")

    if a.gradcam:
        path = save_gradcam(a.image, a.gradcam, model, a.img_size)
        print(f"\n[شكل] تم حفظ Grad-CAM في {path}")


if __name__ == "__main__":
    main()
