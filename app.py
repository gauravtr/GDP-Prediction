"""
app.py
------
Streamlit demo: load the trained model and let a user tweak macroeconomic
indicator values to see the resulting GDP growth forecast. Run with:

    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="India GDP Growth Forecaster", layout="centered")

st.title("🇮🇳 India GDP Growth Forecaster")
st.caption(
    "Random Forest / XGBoost model trained on lagged macroeconomic indicators "
    "(exports, credit growth, investment, industrial output, etc.). "
    "Adjust the sliders to explore scenarios."
)


@st.cache_resource
def load_artifacts():
    feature_cols = joblib.load("models/feature_columns.joblib")
    try:
        model = joblib.load("models/xgb_gdp_model.joblib")
        model_name = "XGBoost"
    except FileNotFoundError:
        model = joblib.load("models/rf_gdp_model.joblib")
        model_name = "Random Forest"
    raw = pd.read_csv("data/india_macro_indicators.csv")
    return model, feature_cols, model_name, raw


model, feature_cols, model_name, raw = load_artifacts()
st.write(f"**Active model:** {model_name}")

st.subheader("Scenario inputs")
inputs = {}
cols = st.columns(2)
for i, feat in enumerate(feature_cols):
    base_col = feat.replace("_lag1", "").replace("_lag2", "")
    series = raw[base_col] if base_col in raw.columns else raw[feat]
    lo, hi, default = float(series.min()), float(series.max()), float(series.iloc[-1])
    with cols[i % 2]:
        if feat == "Crisis_year":
            inputs[feat] = st.selectbox("Crisis year? (external shock flag)", [0, 1], index=int(default))
        else:
            inputs[feat] = st.slider(feat, min_value=round(lo - abs(lo) * 0.2, 2),
                                      max_value=round(hi + abs(hi) * 0.2, 2), value=round(default, 2))

X = pd.DataFrame([inputs])[feature_cols]
pred = model.predict(X)[0]

st.subheader("Forecast")
st.metric("Predicted GDP growth", f"{pred:.2f}%")

with st.expander("Recent historical data"):
    st.dataframe(raw.tail(10).set_index("Year"))
