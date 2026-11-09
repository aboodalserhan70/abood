#!/usr/bin/env bash
# تشغيل كل مراحل المشروع بالترتيب.
# الاستخدام: bash scripts/run_all.sh [backbone]
set -euo pipefail

BACKBONE="${1:-mobilenetv2}"
cd "$(dirname "$0")/.."

echo "==> 1/4  تحليل مخرجات التشغيل الحقيقية (منحنيات + مقاييس + ثقة)"
python scripts/analyze_results.py

echo "==> 2/4  مخطط خط الأنابيب"
python -m src.diagram

echo "==> 3/4  (اختياري) تدريب جديد على مجموعة البيانات الكاملة"
# python -m src.train --epochs 40 --backbone "${BACKBONE}" --img-size 224
# python -m src.evaluate --img-size 224

echo "==> 4/4  بناء نسخة HTML من التقرير"
python scripts/build_report.py

echo
echo "تم. المخرجات في: figures/ · outputs/validation_run/ · report/"
