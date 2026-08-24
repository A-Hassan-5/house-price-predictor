import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="House Price Predictor",
    layout="centered"
)

st.title("House Price Predictor")
st.markdown("""
This app uses the **Random Forest estimator capped with 100 trees and a max depth of 15**.

Adjust the sliders and inputs in the sidebar, then click **Predict Price**.
""")

st.divider()

#load model artifacts
MODEL_PATH = "best_house_price_model.pkl"
SCALER_PATH = "house_price_scaler.pkl"
COLUMNS_PATH = "house_price_feature_columns.pkl"
NEEDS_SCALING_PATH = "house_price_needs_scaling.pkl"


@st.cache_resource
def load_artifacts():
    missing = [p for p in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH, NEEDS_SCALING_PATH]
               if not os.path.exists(p)]
    if missing:
        return None, None, None, None, missing

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
    needs_scaling = joblib.load(NEEDS_SCALING_PATH)
    return model, scaler, feature_columns, needs_scaling, []


model, scaler, feature_columns, needs_scaling, missing_files = load_artifacts()

if missing_files:
    st.error(
        "Missing model artifact file(s): "
        + ", ".join(missing_files)
        + ". Please download these from the Week 1 Colab notebook and place "
          "them in the same folder as this script, then restart the app."
    )
    st.stop()

st.success(f"Model loaded successfully. Using **{type(model).__name__}**.")

# Sidebar 
st.sidebar.header("Enter House Features")
st.sidebar.markdown(
    "Adjust the inputs below to match the property you want to price. "
    "Every field here corresponds exactly to a column the model was trained on."
)

# Reasonable default value + slider range per feature, matched by keyword.
# Anything not matched below falls back to a generic number_input.
FIELD_HINTS = [
    (["bedroom"],            dict(kind="slider", min=0, max=10, value=3, step=1)),
    (["bathroom"],           dict(kind="slider", min=0, max=8, value=2, step=1)),
    (["living area", "sqft_living", "area of the house"],
                              dict(kind="number", min=200, max=15000, value=1800, step=50)),
    (["lot area", "sqft_lot"], dict(kind="number", min=200, max=50000, value=5000, step=100)),
    (["floor"],               dict(kind="slider", min=1, max=4, value=1, step=1)),
    (["waterfront"],          dict(kind="slider", min=0, max=1, value=0, step=1)),
    (["view"],                dict(kind="slider", min=0, max=4, value=0, step=1)),
    (["condition"],           dict(kind="slider", min=1, max=5, value=3, step=1)),
    (["grade"],               dict(kind="slider", min=1, max=13, value=7, step=1)),
    (["basement"],            dict(kind="number", min=0, max=5000, value=0, step=50)),
    (["built year", "year built", "yr_built"],
                              dict(kind="number", min=1900, max=2026, value=2000, step=1)),
    (["renovation", "yr_renovated"],
                              dict(kind="number", min=0, max=2026, value=0, step=1)),
    (["postal", "zip"],       dict(kind="number", min=0, max=999999, value=0, step=1)),
    (["lattitude", "latitude"], dict(kind="float", min=-90.0, max=90.0, value=47.5, step=0.0001)),
    (["longitude"],           dict(kind="float", min=-180.0, max=180.0, value=-122.2, step=0.0001)),
    (["number of schools"],   dict(kind="slider", min=0, max=20, value=5, step=1)),
    (["distance from the airport", "airport"],
                              dict(kind="number", min=0, max=200, value=20, step=1)),
]


def get_hint(col_name: str):
    col_lower = col_name.lower()
    for keywords, hint in FIELD_HINTS:
        if any(kw in col_lower for kw in keywords):
            return hint
    return dict(kind="number", min=0, max=1000000, value=0, step=1)


def pretty_label(col_name: str) -> str:
    return col_name.replace("_", " ").strip().title()


user_input = {}

with st.sidebar.expander("Show exact feature columns expected by the model", expanded=False):
    st.write(feature_columns)

for col in feature_columns:
    hint = get_hint(col)
    label = pretty_label(col)

    if hint["kind"] == "slider":
        user_input[col] = st.sidebar.slider(
            label, int(hint["min"]), int(hint["max"]), int(hint["value"]), step=int(hint["step"])
        )
    elif hint["kind"] == "float":
        user_input[col] = st.sidebar.number_input(
            label, min_value=float(hint["min"]), max_value=float(hint["max"]),
            value=float(hint["value"]), step=float(hint["step"]), format="%.4f"
        )
    else:  # "number"
        user_input[col] = st.sidebar.number_input(
            label, min_value=int(hint["min"]), max_value=int(hint["max"]),
            value=int(hint["value"]), step=int(hint["step"])
        )

# Build the model input row, aligned exactly to the training feature columns
def build_input_row(user_input: dict, feature_columns: list) -> pd.DataFrame:
    row = {col: user_input[col] for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


#predict button
st.divider()

if st.button("Predict Price", type="primary", use_container_width=True):
    input_df = build_input_row(user_input, feature_columns)

    st.subheader("Model Input")
    st.dataframe(input_df, use_container_width=True)

    if needs_scaling:
        input_final = scaler.transform(input_df)
    else:
        input_final = input_df

    prediction = model.predict(input_final)[0]

    st.subheader("Predicted Price")
    st.metric(label="Estimated House Price", value=f"${prediction:,.2f}")

    st.info(
        "This prediction is based on historical patterns learned from the "
        "House Price India dataset and should be used as an estimate only, "
        "not a formal appraisal."
    )
else:
    st.write("Set the property details in the sidebar, then click **Predict Price**.")

st.divider()