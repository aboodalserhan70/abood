#!/usr/bin/env bash
# تشغيل كل مراحل المشروع بالترتيب.
# الاستخدام: bash scripts/run_all.sh [backbone]
set -euo pipefail

BACKBONE="${1:-mobilenetv2}"
cd "$(dirname "$0")/.."

echo "==> 1/6  التحليل الاستكشافي (EDA)"
python -m src.eda

echo "==> 2/6  أشكال المنهجية والبنية"
python -m src.diagram

echo "==> 3/6  خط الأساس الكلاسيكي (SVM)"
python -m src.baseline

echo "==> 4/6  تدريب النموذج (${BACKBONE})"
python -m src.train --epochs 40 --backbone "${BACKBONE}" --img-size 224

echo "==> 5/6  التقييم + Grad-CAM"
python -m src.evaluate --img-size 224

echo "==> 6/6  بناء نسخة HTML من التقرير"
python scripts/build_report.py

echo
echo "تم. المخرجات في: figures/ · outputs/ · report/"
