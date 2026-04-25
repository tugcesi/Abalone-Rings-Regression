"""
app.py - Abalone Rings Predictor (Streamlit)
Run: streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="🐚 Abalone Rings Predictor",
    page_icon="🐚",
    layout="wide",
)

MODEL_PATH    = Path("model.joblib")
FEATURES_PATH = Path("feature_columns.joblib")

SEX_MAP = {'M': 0, 'F': 1, 'I': 2}


@st.cache_resource
def load_artifacts():
    if not all(p.exists() for p in [MODEL_PATH, FEATURES_PATH]):
        return None, None
    model           = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


model, feature_columns = load_artifacts()

if model is None:
    st.error("⚠️ Model dosyaları bulunamadı. Lütfen önce `python save_model.py` çalıştırın.")
    st.stop()

# ── Title ─────────────────────────────────────────────────────────────
st.title("🐚 Abalone Rings Predictor")
st.markdown("Deniz salyangozunun (abalone) fiziksel ölçümlerine göre **halka sayısını (Rings)** tahmin edin.")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("🐚 Ölçüm Değerleri")

sex           = st.sidebar.selectbox("Cinsiyet (Sex)", ["M", "F", "I"],
                                      help="M: Erkek, F: Dişi, I: Yavru")
length        = st.sidebar.slider("Uzunluk (Length)", 0.05, 0.85, 0.55, step=0.005)
diameter      = st.sidebar.slider("Çap (Diameter)", 0.05, 0.70, 0.43, step=0.005)
height        = st.sidebar.slider("Yükseklik (Height)", 0.00, 0.30, 0.15, step=0.005)
whole_weight  = st.sidebar.slider("Toplam Ağırlık (Whole weight)", 0.002, 3.0, 0.80, step=0.001)
shucked       = st.sidebar.slider("İç Et Ağırlığı (Whole weight.1)", 0.001, 1.6, 0.36, step=0.001)
viscera       = st.sidebar.slider("İç Organ Ağırlığı (Whole weight.2)", 0.001, 0.80, 0.18, step=0.001)
shell_weight  = st.sidebar.slider("Kabuk Ağırlığı (Shell weight)", 0.001, 1.1, 0.24, step=0.001)

predict_btn = st.sidebar.button("🔍 Tahmin Et", use_container_width=True)


# ── Build Input ────────────────────────────────────────────────────────
def build_input_df() -> pd.DataFrame:
    volume      = length * diameter * height
    meat_weight = whole_weight - shell_weight

    row = {
        'Shell weight':   shell_weight,
        'Height':         height,
        'Diameter':       diameter,
        'Volume':         volume,
        'Length':         length,
        'Whole weight':   whole_weight,
        'Whole weight.2': viscera,
        'Meat_weight':    meat_weight,
        'Whole weight.1': shucked,
        'Sex':            SEX_MAP[sex],
    }
    return pd.DataFrame([row])[feature_columns].astype(float)


# ── Prediction ─────────────────────────────────────────────────────────
if predict_btn:
    X_input    = build_input_df()
    prediction = float(model.predict(X_input)[0])
    rings_int  = max(1, round(prediction))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.success(f"🐚 Tahmini Halka Sayısı: **{prediction:.2f}**  →  **~{rings_int} halka**")
        st.markdown(f"📅 Yaklaşık Yaş: **{rings_int + 1.5:.1f} yıl**  *(halka sayısı + 1.5)*")

        # Mühendislik özellikleri
        volume     = length * diameter * height
        meat_w     = whole_weight - shell_weight
        shell_r    = shell_weight / (whole_weight + 1)
        density    = whole_weight / (volume + 1)

        st.markdown("### 📋 Hesaplanan Özellikler")
        eng_df = pd.DataFrame({
            "Özellik":  ["Volume", "Meat_weight", "Shell_ratio", "Density"],
            "Değer":    [f"{volume:.4f}", f"{meat_w:.4f}", f"{shell_r:.4f}", f"{density:.4f}"],
            "Açıklama": [
                "Uzunluk × Çap × Yükseklik",
                "Toplam - Kabuk Ağırlığı",
                "Kabuk / (Toplam + 1)",
                "Toplam / (Hacim + 1)",
            ]
        })
        st.dataframe(eng_df, use_container_width=True, hide_index=True)

    with col2:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={"text": "Tahmin Edilen Rings"},
            gauge={
                "axis":  {"range": [0, 30]},
                "bar":   {"color": "royalblue"},
                "steps": [
                    {"range": [0,  8],  "color": "lightcyan"},
                    {"range": [8,  15], "color": "lightblue"},
                    {"range": [15, 30], "color": "steelblue"},
                ],
                "threshold": {
                    "line":      {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value":     prediction,
                },
            },
        ))
        fig_gauge.update_layout(height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Feature Importance
    st.markdown("### 📊 Özellik Önemi")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi_df = (
            pd.DataFrame({"Özellik": feature_columns, "Önem": importances})
            .sort_values("Önem", ascending=True)
        )
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Önem"],
            y=fi_df["Özellik"],
            orientation="h",
            marker_color="steelblue",
        ))
        fig_fi.update_layout(
            title="Özellik Önemi (LightGBM)",
            xaxis_title="Önem Skoru",
            yaxis_title="Özellik",
            height=400,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

else:
    st.info("👈 Sol panelden ölçüm değerlerini girin ve **🔍 Tahmin Et** butonuna tıklayın.")

# ── About ──────────────────────────────────────────────────────────────
with st.expander("ℹ️ Proje Hakkında"):
    st.markdown("""
    ### 🐚 Abalone Rings Predictor

    Bu uygulama **Kaggle Playground Series S4E4** yarışması için geliştirilen
    regresyon modelini kullanarak abalone (deniz salyangozu) halka sayısını tahmin eder.
    Halka sayısı, abaloneun yaşını belirlemek için kullanılır: **Yaş = Rings + 1.5**

    **Orijinal Özellikler:**
    | Özellik | Açıklama |
    |---|---|
    | Sex | M: Erkek, F: Dişi, I: Yavru |
    | Length | Kabuk uzunluğu (mm) |
    | Diameter | Kabuk çapı (mm) |
    | Height | Kabuk yüksekliği (mm) |
    | Whole weight | Toplam ağırlık (g) |
    | Whole weight.1 | İç et ağırlığı (g) |
    | Whole weight.2 | İç organ ağırlığı (g) |
    | Shell weight | Kabuk ağırlığı (g) |

    **Mühendislik Özellikleri:**
    - **Volume** = Length × Diameter × Height
    - **Meat_weight** = Whole weight − Shell weight
    - **Shell_ratio** = Shell weight / (Whole weight + 1)
    - **Density** = Whole weight / (Volume + 1)

    **Model:** LightGBM Regressor  
    **Veri Kaynağı:** [Kaggle – Playground Series S4E4](https://www.kaggle.com/competitions/playground-series-s4e4)
    """)