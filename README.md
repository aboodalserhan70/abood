# 🌽 كشف أمراض أوراق الذرة — Corn Leaf Disease Detection

مشروع رؤية حاسوبية لتصنيف أوراق الذرة إلى أربع فئات: **اللفحة الشمالية (Blight)**،
**الصدأ الشائع (Common Rust)**، **التبقّع الرمادي (Gray Leaf Spot)**، و**الورقة السليمة (Healthy)**.

> **التقرير الكامل بالعربية:** [`report/corn_leaf_disease_report_ar.md`](report/corn_leaf_disease_report_ar.md)
> · نسخة HTML جاهزة للطباعة: [`report/corn_leaf_disease_report_ar.html`](report/corn_leaf_disease_report_ar.html)

---

## النتائج

| المقياس | القيمة |
|---|---:|
| الدقة على مجموعة التحقّق | **93.79%** (95% CI: 91.9–95.2%) |
| Macro-Precision / Recall / F1 | 0.9289 / 0.8925 / **0.9063** |
| Weighted-F1 | 0.9361 |
| صور التحقّق | 837 (من 4,184 صورة بعد التنظيف) |
| النموذج المختار | MobileNetV2 مُجمّدة + رأس خطّي (5,124 معاملاً قابلاً للتدريب) |
| حجم النموذج | 2,263,108 معاملاً · 9.24 MiB |

**لكل فئة:**

| الفئة | Precision | Recall | F1 | الدعم |
|---|---:|---:|---:|---:|
| Blight | 0.864 | 0.953 | 0.906 | 253 |
| Common_Rust | 0.977 | 0.958 | 0.968 | 265 |
| **Gray_Leaf_Spot** | 0.875 | **0.659** | **0.752** | 85 |
| Healthy | 1.000 | 1.000 | 1.000 | 234 |

نقطتان تستحقّان الانتباه: **54% من كل الأخطاء** هي تصنيف Gray_Leaf_Spot على أنها Blight،
ولا توجد **أي** حالة خلط بين «سليم» و«مريض» في الاتجاهين. وبإحالة 10% من الصور
(ثقة < 0.70) إلى مراجعة بشرية ترتفع دقة القرارات المقبولة آلياً إلى **96.4%**.

> ⚠️ الأرقام محسوبة على مجموعة التحقّق نفسها المستخدمة لاختيار النموذج
> (لا توجد مجموعة اختبار مستقلة، والتقسيم عشوائي غير طبقي). التفاصيل في القسم 7 من التقرير.

---

## الفئات

| الفئة | الاسم العلمي للمسبّب | صور المجموعة |
|---|---|---:|
| Blight — اللفحة الشمالية | *Exserohilum turcicum* | 1,144 |
| Common_Rust — الصدأ الشائع | *Puccinia sorghi* | 1,306 |
| Gray_Leaf_Spot — التبقّع الرمادي | *Cercospora zeae-maydis* | 572 |
| Healthy — سليم | — | 1,162 |

يقبل الكود أسماء المجلدات الأصلية (`Blight`، `Common_Rust`، `Gray_Leaf_Spot`، `Healthy`)
وأيضاً الأسماء المسبوقة بـ `Corn_` (كما في `data/raw/`).

---

## البنية

```
.
├── data/raw/<class>/           عيّنة صور (18 صورة) لأغراض العرض
├── src/
│   ├── config.py               الفئات، الأسماء البديلة، المسارات، المعاملات
│   ├── data.py                 الفحص، التقسيم الطبقي، خطوط tf.data، التوسيع
│   ├── model.py                CNN من الصفر + تعلّم نقل (MobileNetV2 / EfficientNetB0)
│   ├── train.py                تدريب بمرحلتين (تجميد ثم ضبط دقيق)
│   ├── evaluate.py             المقاييس، مصفوفة الالتباس، معرض التوقّعات
│   ├── gradcam.py              خرائط Grad-CAM (تفسير بصري)
│   ├── predict.py              توقّع صورة واحدة من سطر الأوامر
│   ├── eda.py                  تحليل استكشافي للمجموعة
│   ├── baseline.py             خط أساس كلاسيكي اختياري (ميزات يدوية + SVM)
│   └── diagram.py              مخطط خط الأنابيب
├── outputs/validation_run/     ← مخرجات التشغيل الحقيقية (837 توقّعاً، مقاييس، manifests)
├── figures/                    جميع أشكال التقرير
├── report/                     التقرير (Markdown + HTML مستقل)
└── scripts/
    ├── analyze_results.py      منحنيات التدريب + مقاييس الفئات + تحليل الثقة
    ├── build_report.py         بناء نسخة HTML من التقرير
    └── run_all.sh              تشغيل كل المراحل
```

---

## التشغيل

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# إعادة توليد الأشكال التحليلية من مخرجات التشغيل الحقيقية
python scripts/analyze_results.py
python scripts/build_report.py          # نسخة HTML مستقلة (الصور مضمّنة)

# تدريب جديد على مجموعة بيانات كاملة
python -m src.train --epochs 40 --backbone mobilenetv2 --img-size 224
python -m src.evaluate --img-size 224

# توقّع صورة واحدة
python -m src.predict --image leaf.jpg --gradcam figures/leaf_cam.png
```

> تحميل أوزان ImageNet يحتاج اتصالاً بالإنترنت في أول تشغيل؛ وعند عدم توفّره
> يبني الكود الشبكة بأوزان عشوائية ويحذّر بذلك صراحةً.

---

## المتطلبات

Python 3.10+ · TensorFlow 2.16+ · scikit-learn · matplotlib · Pillow · markdown.
للتفاصيل: [`requirements.txt`](requirements.txt).
