"""
بناء النماذج: شبكة سي إن إن من الصفر، أو تعلّم نقل (Transfer Learning).

ملاحظة مهمة: تحميل أوزان ImageNet يتطلب إنترنت. في بيئة بلا إنترنت
سيتم بناء الشبكة الأساسية بأوزان عشوائية تلقائياً مع تحذير واضح.
"""
from __future__ import annotations

import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="corn")
class SparseCategoricalCrossentropyLS(tf.keras.losses.Loss):
    """خسارة تصنيف متعدد مع تنعيم التسميات (label smoothing) لتسميات صحيحة كأعداد."""

    def __init__(self, smoothing: float = 0.05, name: str = "sparse_ce_ls", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smoothing = float(smoothing)
        self._ce = tf.keras.losses.CategoricalCrossentropy()

    def call(self, y_true, y_pred):
        n_classes = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), n_classes)
        per_class = self.smoothing / tf.cast(n_classes, tf.float32)
        y_true = y_true * (1.0 - self.smoothing) + per_class
        return self._ce(y_true, y_pred)

    def get_config(self):
        base = super().get_config()
        base.update({"smoothing": self.smoothing})
        return base


BACKBONES = {
    "mobilenetv2": tf.keras.applications.MobileNetV2,
    "efficientnetb0": tf.keras.applications.EfficientNetB0,
}


def _pretrained_backbone(name: str, img_size: int, try_weights: bool = True):
    """يحاول تحميل أوزان ImageNet، ويعود إلى أوزان عشوائية عند الفشل."""
    builder = BACKBONES[name]
    if try_weights:
        try:
            base = builder(
                include_top=False, weights="imagenet",
                input_shape=(img_size, img_size, 3),
            )
            print(f"[معلومة] تم تحميل أوزان ImageNet لـ {name}.")
            return base, True
        except Exception as exc:  # noqa: BLE001
            print(f"[تحذير] تعذّر تحميل أوزان ImageNet ({exc}). "
                  f"سيتم البناء بأوزان عشوائية — النتائج لن تكون ذات معنى علمي.")
    base = builder(include_top=False, weights=None, input_shape=(img_size, img_size, 3))
    return base, False


def build_model(num_classes: int, img_size: int = 224, backbone: str = "scratch",
                dropout: float = 0.3) -> tf.keras.Model:
    """يبني النموذج ويعيده مع الشبكة الأساسية (إن وُجدت) للضبط الدقيق."""
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="leaf_image")

    # تطبيع داخلي: يحوّل [0,1] إلى المدى الذي تتوقعه الشبكة الأساسية
    if backbone in BACKBONES:
        base, loaded = _pretrained_backbone(backbone, img_size)
        x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="to_minus1_1")(inputs)
        if backbone == "efficientnetb0":
            x = tf.keras.layers.Rescaling(127.5, offset=127.5, name="to_0_255")(inputs)
        x = base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = tf.keras.layers.BatchNormalization(name="bn_head")(x)
        x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    else:
        # شبكة خفيفة من الصفر — مناسبة للصور صغيرة الحجم وللتجارب السريعة
        x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = tf.keras.layers.Dropout(dropout, name="dropout")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name=f"corn_{backbone}")
    return model


def unfreeze_top(model: tf.keras.Model, n_layers: int = 30) -> None:
    """يفكّ تجميد آخر n طبقة من الشبكة الأساسية للضبط الدقيق."""
    base = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    if base is None:
        return
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    # طبقات BatchNormalization تبقى مجمّدة لتجنّب تدهور الإحصاءات
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    print(f"[معلومة] تم فك تجميد آخر {n_layers} طبقة من الشبكة الأساسية.")


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3,
                  label_smoothing: float = 0.0) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=1e-4
        ),
        loss=SparseCategoricalCrossentropyLS(label_smoothing),
        # تُحسب الدقة/الاستدعاء/F1 لكل فئة في src/evaluate.py عبر scikit-learn
        metrics=["accuracy"],
    )
    return model


def count_params(model: tf.keras.Model) -> dict[str, int]:
    trainable = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    total = int(model.count_params())
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
