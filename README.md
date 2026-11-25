# Phishing Link Detector

## تشغيل سريع
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py          # الواجهة (شغالة لوحدها، مش محتاجة API)
# أو الـ API:
uvicorn main:app --port 8001            # POST /predict  {"url": "..."}
```

## الملفات
| ملف | الوظيفة |
|---|---|
| `feature_extraction.py` | استخراج 100 فيتشر من اللينك (بدون tldextract، أوفلاين) |
| `build_dataset.py` | يبني `Dataset_expanded.csv` (الداتا الأصلية + لينكات مولّدة متنوعة) |
| `train.py` | يدرّب الموديل ويحفظ `Final_model.pkl` و `model_scaler.pkl` |
| `predictor.py` | تحميل الموديل والتصنيف (مشترك بين الـ API والواجهة) |
| `main.py` / `streamlit_app.py` | الـ API والواجهة |
| `_old_version/` | نسخة الملفات القديمة للمرجع |

لإعادة التدريب: `python build_dataset.py && python train.py`
(لو غيّرت نسخة scikit-learn أعد التدريب عشان ملفات pkl تتوافق).

## تنبيه مهم
مفيش موديل يقدر يتوقع "كل لينكات الدنيا" بدقة 100%. الموديل بيحكم على شكل اللينك فقط
(مش بيفتح الموقع ولا بيشيك على blacklist)، فالأفضل تضيف عليه فحص سمعة الدومين لو هتستخدمه في الإنتاج.
