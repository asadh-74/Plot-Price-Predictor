"""Tiny demo UI for the API. Run: streamlit run app/streamlit_demo.py"""
import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Plot Price Predictor", page_icon="🏞️")
st.title("🏞️ Plot Price Predictor")
st.caption("ML + Deep Learning demo — FastAPI backend, PyTorch MLP recommended model")

with st.sidebar:
    st.header("Plot features")
    acres = st.number_input("Acres", 0.05, 60.0, 2.5, 0.05)
    location = st.selectbox("Location", ["Downtown", "Riverside", "Lakeview", "Old Town",
                                         "Meadowbrook", "Hillcrest", "Greenfield", "Industrial Zone"])
    zoning = st.selectbox("Zoning", ["residential", "commercial", "agricultural"])
    road = st.selectbox("Road access", ["paved", "gravel", "dirt"])
    corner = st.checkbox("Corner plot")
    dist = st.slider("Distance to city (km)", 0.2, 80.0, 8.0)
    front = st.number_input("Frontage (m)", 4.0, 80.0, 22.0)

if st.button("Estimate price", type="primary"):
    payload = {"acres": acres, "location": location, "zoning": zoning,
               "road_access": road, "corner_plot": corner,
               "distance_to_city_km": dist, "frontage_m": front}
    r = requests.post(f"{API}/predict", json=payload, timeout=15)
    if r.status_code == 200:
        d = r.json()
        st.metric("Estimated price", f"${d['estimated_price']:,.0f}",
                  delta=f"range ${d['price_range']['low']:,.0f} – ${d['price_range']['high']:,.0f}")
        st.subheader("Model comparison")
        st.bar_chart({k: v for k, v in d["predictions"].items()})
    else:
        st.error(f"API error {r.status_code}: {r.text}")
