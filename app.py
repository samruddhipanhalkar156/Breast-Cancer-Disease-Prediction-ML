# app.py
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# -------------------------
# Paths
# -------------------------
MODELS_DIR = Path("Models")
RESULTS_DIR = Path("Model Results")

# Load results
results_files = list(RESULTS_DIR.glob("model_results_*.xlsx"))
metrics_df = pd.read_excel(results_files[-1]) if results_files else pd.DataFrame()

# Load scaler
scaler_path = MODELS_DIR / "scaler.pkl"
if not scaler_path.exists():
    st.error(" scaler.pkl not found! Please run main.py first.")
    st.stop()
else:
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

# Models available
MODEL_PATHS = {
    "Logistic Regression": MODELS_DIR / "LogisticRegression_binary.pkl",
    "SVM": MODELS_DIR / "SVM_binary.pkl",
    "Random Forest": MODELS_DIR / "RandomForest_binary.pkl",
    "XGBoost": MODELS_DIR / "XGBoost_binary.pkl",
}

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="🩺 Breast Cancer Prediction", layout="wide")

st.title("🩺 Breast Cancer Diagnostic Prediction System")

# -------------------------
# Model selection
# -------------------------
model_choice = st.selectbox("Choose a trained model:", list(MODEL_PATHS.keys()))

# Show metrics
if not metrics_df.empty:
    st.write("### Model Metrics")
    binary_metrics = metrics_df[
        (metrics_df["Model"] == model_choice.replace(" ", "")) &
        (metrics_df["Task"] == "binary")
    ]
    st.dataframe(binary_metrics)

# Load model
with open(MODEL_PATHS[model_choice], "rb") as f:
    model = pickle.load(f)

# -------------------------
# Single Input Prediction
# -------------------------
st.subheader("🔹 Single Patient Prediction")

# Features actually used in the model
feature_names = [
    "radius1 (Mean Radius)", 
    "texture1 (Mean Texture)", 
    "smoothness1 (Mean Smoothness)", 
    "compactness1 (Mean Compactness)", 
    "symmetry1 (Mean Symmetry)", 
    "fractal_dimension1 (Mean Fractal Dimension)",

    "radius2 (SE Radius)", 
    "texture2 (SE Texture)", 
    "smoothness2 (SE Smoothness)", 
    "compactness2 (SE Compactness)", 
    "concavity2 (SE Concavity)", 
    "symmetry2 (SE Symmetry)", 
    "fractal_dimension2 (SE Fractal Dimension)",

    "radius3 (Worst Radius)", 
    "smoothness3 (Worst Smoothness)", 
    "compactness3 (Worst Compactness)", 
    "concavity3 (Worst Concavity)", 
    "symmetry3 (Worst Symmetry)", 
    "fractal_dimension3 (Worst Fractal Dimension)"
]

inputs = {}
cols = st.columns(3)
for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        val = st.text_input(feature, "0", key=f"feat_{i}")
        try:
            val = float(val)
        except:
            val = 0.0
        inputs[feature.split()[0]] = val  # use raw column name (before space)

if st.button("Predict Breast Cancer"):
    single_df = pd.DataFrame([inputs])
    single_scaled = scaler.transform(single_df)

    pred = model.predict(single_scaled)[0]
    if pred == 0:
        st.success("✅ Prediction: **Benign (Non-Cancerous)**")
    else:
        st.error("⚠️ Prediction: **Malignant (Cancerous)**")

# -------------------------
# Batch Prediction
# -------------------------
st.subheader("📂 Batch Prediction via CSV")
uploaded_file = st.file_uploader("Upload CSV with features", type=["csv"])

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(user_df.head())

    scaled = scaler.transform(user_df)
    preds = model.predict(scaled)

    user_df["Prediction"] = ["Benign" if p == 0 else "Malignant" for p in preds]

    st.write("### Predictions")
    st.dataframe(user_df.head())

    csv = user_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

# -------------------------
# Footer
# -------------------------
st.markdown("---", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align: center;">
        <p><strong>Created by: Samruddhi R. Panhalkar</strong></p>
        <p><strong>Roll No: USN- 2MM22RI014</strong></p>
        <p>📧 samruddhipanhalkar156@gmail.com | 📱 +91-8951831491</p>
        <p>🏫 Maratha Mandal Engineering College</p>
    </div>
    """,
    unsafe_allow_html=True,
)

