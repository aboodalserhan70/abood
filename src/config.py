"""
الإعدادات العامة للمشروع.
Central configuration for the corn leaf disease detection project.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# المسارات | Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

# ---------------------------------------------------------------------------
# الفئات | Classes
# اسم المجلد في مجموعة البيانات  ->  (الاسم اللاتيني، الاسم العربي، الاسم العلمي للمرض)
# ---------------------------------------------------------------------------
CLASS_INFO: dict[str, dict[str, str]] = {
    "Corn_Blight": {
        "en": "Northern Leaf Blight",
        "ar": "اللفحة الشمالية",
        "latin": "Exserohilum turcicum",
        "color": "#C0392B",
    },
    "Corn_Common_Rust": {
        "en": "Common Rust",
        "ar": "الصدأ الشائع",
        "latin": "Puccinia sorghi",
        "color": "#D35400",
    },
    "Corn_Gray_Spot": {
        "en": "Gray Leaf Spot",
        "ar": "التبقع الرمادي",
        "latin": "Cercospora zeae-maydis",
        "color": "#7F8C8D",
    },
    "Corn_Health": {
        "en": "Healthy",
        "ar": "سليم",
        "latin": "-",
        "color": "#27AE60",
    },
}

CLASS_NAMES: list[str] = list(CLASS_INFO.keys())

# أسماء بديلة للمجلدات: تسمح باستخدام أسماء المجلدات الأصلية للمجموعة
# (Blight / Common_Rust / Gray_Leaf_Spot / Healthy) أو الأسماء المسبوقة بـ Corn_
CLASS_ALIASES: dict[str, str] = {
    "Corn_Blight": "Corn_Blight",
    "Blight": "Corn_Blight",
    "Corn_Common_Rust": "Corn_Common_Rust",
    "Common_Rust": "Corn_Common_Rust",
    "Corn_Gray_Spot": "Corn_Gray_Spot",
    "Gray_Leaf_Spot": "Corn_Gray_Spot",
    "Corn_Health": "Corn_Health",
    "Healthy": "Corn_Health",
}


# ---------------------------------------------------------------------------
# معاملات النموذج والتدريب | Model / training hyper-parameters
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    data_dir: Path = DATA_DIR
    img_size: int = 224          # دقة الإدخال (بكسل)
    batch_size: int = 16
    epochs: int = 40             # مرحلة أولى (تجميد الشبكة الأساسية)
    fine_tune_epochs: int = 15   # مرحلة الضبط الدقيق
    learning_rate: float = 1e-3
    fine_tune_lr: float = 1e-5
    weight_decay: float = 1e-4
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    backbone: str = "scratch"    # scratch | mobilenetv2 | efficientnetb0
    label_smoothing: float = 0.05
    dropout: float = 0.3
    use_class_weights: bool = True
    output_dir: Path = OUTPUT_DIR
    figures_dir: Path = FIGURES_DIR
    num_parallel_calls: int = 2
    cache: bool = False


def ensure_dirs() -> None:
    """إنشاء المجلدات الناتجة إن لم تكن موجودة."""
    for path in (OUTPUT_DIR, FIGURES_DIR, MODELS_DIR, OUTPUT_DIR / "logs"):
        os.makedirs(path, exist_ok=True)
