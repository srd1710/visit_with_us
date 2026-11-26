"""Streamlit app for predicting Wellness Tourism Package purchase likelihood."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.joblib"
METADATA_PATH = HERE / "model_metadata.json"

st.set_page_config(page_title="Visit with Us | Wellness Package Predictor", page_icon="✈️", layout="wide")
st.title("✈️ Visit with Us")
st.subheader("Wellness Tourism Package — Purchase Prediction")
st.write("Enter a customer's details to estimate the likelihood that they will purchase the Wellness Tourism Package.")

@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "The trained model is not available yet. Run the GitHub Actions training pipeline "
            "and ensure model.joblib and model_metadata.json are committed to this folder."
        )
    return joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text())

try:
    model, metadata = load_assets()
except Exception as exc:
    st.error(str(exc))
    st.stop()

features = metadata["features"]
with st.form("customer_form"):
    st.markdown("### Customer information")
    cols = st.columns(2)
    values = {}
    for index, (name, spec) in enumerate(features.items()):
        with cols[index % 2]:
            label = name.replace("_", " ")
            if spec["type"] == "category":
                options = spec.get("options", [])
                default = spec.get("default", options[0] if options else "")
                if options:
                    selected_index = options.index(default) if default in options else 0
                    values[name] = st.selectbox(label, options, index=selected_index)
                else:
                    values[name] = st.text_input(label, value=str(default))
            else:
                low, high = float(spec["min"]), float(spec["max"])
                default = float(spec["default"])
                if low == high:
                    values[name] = st.number_input(label, value=default)
                else:
                    # number_input handles continuous and integer-valued source columns.
                    step = 1.0 if float(low).is_integer() and float(high).is_integer() else 0.5
                    values[name] = st.number_input(
                        label, min_value=low, max_value=high,
                        value=min(max(default, low), high), step=step
                    )
    submitted = st.form_submit_button("Predict purchase likelihood", type="primary")

if submitted:
    input_df = pd.DataFrame([values], columns=list(features.keys()))
    probability = float(model.predict_proba(input_df)[0][1])
    prediction = int(probability >= 0.5)
    st.divider()
    if prediction:
        st.success("Prediction: Likely to purchase the package")
    else:
        st.info("Prediction: Unlikely to purchase the package")
    st.metric("Estimated purchase probability", f"{probability:.1%}")
    st.progress(probability)
    st.caption("This is a model estimate, not a guarantee of customer behavior.")

with st.expander("Model evaluation"):
    st.json(metadata.get("metrics", {}))
