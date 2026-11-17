from pathlib import Path
import streamlit as st
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent
package = joblib.load(BASE_DIR / "deployment_model" / "clustering_deployment.joblib")
model = package["model"]
scaler = package["scaler"]
feature_columns = package["feature_columns"]

st.title("Segmentasi Pelanggan Online Retail")
st.write("Prediksi cluster pelanggan menggunakan K-Means")

recency = st.number_input("Recency (hari)", min_value=0, value=15)
frequency = st.number_input("Frequency (transaksi)", min_value=1, value=8)
monetary = st.number_input("Monetary (GBP)", min_value=1.0, value=2500.0)

if st.button("Prediksi Cluster"):
    data = pd.DataFrame({
        "Recency": [recency],
        "Frequency_log": [np.log1p(frequency)],
        "Monetary_log": [np.log1p(monetary)]
    })
    cluster = model.predict(scaler.transform(data[feature_columns]))
    st.success(f"Pelanggan masuk ke Cluster {cluster[0]}")
