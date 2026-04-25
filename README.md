# 🐚 Abalone Rings Predictor

Abalone (deniz salyangozu) fiziksel ölçümlerinden **halka sayısını (Rings)** tahmin eden makine öğrenmesi uygulaması.

> 📅 **Yaş hesabı:** Rings + 1.5 yıl

---

## 📊 Proje Hakkında

**Kaggle Playground Series S4E4** yarışması için geliştirilmiştir.

**Orijinal Özellikler:**
| Özellik | Açıklama |
|---|---|
| Sex | M: Erkek, F: Dişi, I: Yavru |
| Length | Kabuk uzunluğu |
| Diameter | Kabuk çapı |
| Height | Kabuk yüksekliği |
| Whole weight | Toplam ağırlık |
| Whole weight.1 | İç et ağırlığı |
| Whole weight.2 | İç organ ağırlığı |
| Shell weight | Kabuk ağırlığı |

**Mühendislik Özellikleri:**
- `Volume` = Length × Diameter × Height
- `Meat_weight` = Whole weight − Shell weight
- `Shell_ratio` = Shell weight / (Whole weight + 1)
- `Density` = Whole weight / (Volume + 1)

---

## 🚀 Kurulum ve Kullanım

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. Modeli eğit
python save_model.py

# 3. Uygulamayı başlat
streamlit run app.py
```

---

## 🛠️ Kullanılan Teknikler

- **Feature Engineering** – Volume, Meat_weight, Shell_ratio, Density
- **LightGBM Regressor** – Hızlı gradient boosting regresyonu
- **Sex Encoding** – M→0, F→1, I→2

---

## 📁 Proje Yapısı

```
├── train.csv                                   # Eğitim verisi
├── test.csv                                    # Test verisi
├── regression-with-an-abalone-dataset.ipynb    # Analiz notebook'u
├── save_model.py                               # Model eğitimi
├── app.py                                      # Streamlit uygulaması
├── requirements.txt                            # Bağımlılıklar
├── model.joblib                                # Eğitilmiş model
└── feature_columns.joblib                      # Özellik sırası
```

---

**Veri Kaynağı:** [Kaggle Playground Series S4E4](https://www.kaggle.com/competitions/playground-series-s4e4)
