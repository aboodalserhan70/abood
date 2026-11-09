# 🌽 كشف أمراض أوراق الذرة — Corn Leaf Disease Detection

مشروع رؤية حاسوبية متكامل لتصنيف أوراق الذرة إلى أربع فئات: **اللفحة الشمالية**،
**الصدأ الشائع**، **التبقّع الرمادي**، و**الورقة السليمة** — من صورة واحدة.

> **التقرير الكامل بالعربية:** [`report/corn_leaf_disease_report_ar.md`](report/corn_leaf_disease_report_ar.md)
> · نسخة HTML جاهزة للطباعة: [`report/corn_leaf_disease_report_ar.html`](report/corn_leaf_disease_report_ar.html)

---

## الفئات الأربع

| المجلد | الفئة | الاسم العلمي للمسبّب | صور العيّنة |
|---|---|---|---:|
| `Corn_Blight` | اللفحة الشمالية — Northern Leaf Blight | *Exserohilum turcicum* | 5 |
| `Corn_Common_Rust` | الصدأ الشائع — Common Rust | *Puccinia sorghi* | 3 |
| `Corn_Gray_Spot` | التبقّع الرمادي — Gray Leaf Spot | *Cercospora zeae-maydis* | 5 |
| `Corn_Health` | سليم — Healthy | — | 5 |

---

## البنية

```
.
├── data/raw/<class>/        صور مجموعة البيانات (مجلد لكل فئة)
├── src/
│   ├── config.py            الفئات، المسارات، المعاملات الفائقة
│   ├── data.py              الفحص، التقسيم الطبقي، خطوط tf.data، التوسيع
│   ├── model.py             CNN من الصفر + تعلّم نقل (MobileNetV2 / EfficientNetB0)
│   ├── train.py             تدريب بمرحلتين (تجميد ثم ضبط دقيق)
│   ├── evaluate.py          المقاييس، مصفوفة الالتباس، معرض التوقّعات
│   ├── gradcam.py           خرائط Grad-CAM
│   ├── predict.py           توقّع صورة واحدة من سطر الأوامر
│   ├── eda.py               التحليل الاستكشافي والأشكال
│   ├── baseline.py          خط أساس كلاسيكي (ميزات يدوية + SVM)
│   └── diagram.py           أشكال المنهجية والبنية
├── figures/                 جميع أشكال التقرير (تُولَّد آلياً)
├── outputs/                 metrics.json · history.json · النماذج المدرَّبة
├── report/                  التقرير (Markdown + HTML)
└── scripts/
    ├── run_all.sh           تشغيل كل المراحل بالترتيب
    └── build_report.py      بناء نسخة HTML من التقرير
```

---

## التشغيل السريع

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) التحليل الاستكشافي + أشكال المنهجية
python -m src.eda
python -m src.diagram

# 2) خط الأساس الكلاسيكي (SVM على ميزات لونية ونسيجية)
python -m src.baseline

# 3) تدريب الشبكة العصبية (تعلّم نقل)
python -m src.train --epochs 40 --backbone mobilenetv2 --img-size 224

# 4) التقييم الكامل + خرائط Grad-CAM
python -m src.evaluate --img-size 224

# 5) توقّع صورة واحدة
python -m src.predict --image data/raw/Corn_Blight/leaf.jpg \
                      --gradcam figures/my_leaf_gradcam.png

# 6) إعادة بناء نسخة HTML من التقرير
python scripts/build_report.py
```

أو ببساطة: `bash scripts/run_all.sh`

> **ملاحظة:** تحميل أوزان ImageNet يحتاج اتصالاً بالإنترنت في أول تشغيل.
> عند عدم توفّره يبني الكود الشبكة بأوزان عشوائية ويحذّر بذلك صراحةً.

---

## النتائج الحالية

| التجربة | البروتوكول | الدقة | Macro-F1 | ملاحظة |
|---|---|---:|---:|---|
| خط أساس SVM + 77 ميزة يدوية | Leave-One-Out على 18 صورة | **72.2%** | 0.76 | نتيجة حقيقية على العيّنة المتاحة |
| CNN من الصفر (96×96) | 10 تدريب / 4 اختبار | 25.0% | 0.17 | **تشغيل تحقّق فقط** — العيّنة أصغر من أن تُتعلَّم |
| MobileNetV2 + تعلّم نقل | 70/15/15 على المجموعة الكاملة | — | — | يُنفَّذ بعد إضافة بقية الصور |

خط الأساس يكشف أن الفئات قابلة للفصل جزئياً بميزات بسيطة، وأن **اللفحة الشمالية
والتبقّع الرمادي** هما الفئتان الأكثر تداخلاً — وهو ما تؤكّده مصفوفة الالتباس
وإسقاط الميزات (PCA / t-SNE).

---

## إضافة بقية مجموعة البيانات

يكفي وضع الصور في المجلدات الأربعة داخل `data/raw/` بأسماء الفئات نفسها،
ثم تشغيل:

```bash
python -m src.train --epochs 40 --backbone mobilenetv2 --img-size 224
python -m src.evaluate --img-size 224
```

سيُعيد المشروع توليد جميع الأشكال والملفات في `figures/` و`outputs/` تلقائياً،
ويمكن بعدها تحديث جداول النتائج في التقرير.

---

## المتطلبات

Python 3.10+ · TensorFlow 2.16 · scikit-learn · matplotlib · Pillow.
للتفاصيل: [`requirements.txt`](requirements.txt).

## الإسناد

مجموعة البيانات المستخدمة في التقرير مجموع صور لأوراق ذرة بأربع فئات.
الأعمال المرجعية الأساسية مذكورة في قسم المراجع داخل التقرير، وأهمها
PlantVillage (Hughes & Salathé, 2015) و Mohanty et al. (2016).
