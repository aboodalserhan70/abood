"""
Grad-CAM: خريطة التنشيط المتدرّجة لتفسير قرار النموذج بصرياً.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def _find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, tf.keras.Model):  # شبكة أساسية متداخلة
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    raise ValueError("لم يتم العثور على طبقة Conv2D لاستخدامها في Grad-CAM.")


def compute_gradcam(model: tf.keras.Model, image: np.ndarray,
                    class_index: int | None = None,
                    layer_name: str | None = None) -> np.ndarray:
    """
    image: مصفوفة بالشكل (1, H, W, 3) ومداها [0, 1].
    يعيد خريطة تشابه بالشكل (H, W) ومداها [0, 1].
    """
    layer_name = layer_name or _find_last_conv_layer(model)
    target_layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model(model.inputs, [target_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(image)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))           # أوزان القنوات
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0.0)                       # ReLU
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """يراكب خريطة الحرارة على الصورة الأصلية (كلاهما بمدى [0,1])."""
    import matplotlib.cm as cm

    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 4:
        image = image[0]
    heat = tf.image.resize(heatmap[..., np.newaxis], image.shape[:2]).numpy()[..., 0]
    heat = np.clip(heat, 0.0, 1.0)
    colored = cm.jet(heat)[..., :3].astype(np.float32)   # خرائط لونية ناعمة
    return np.clip((1 - alpha) * image + alpha * colored, 0, 1)
