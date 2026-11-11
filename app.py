
import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="SLA Risk Simulator", page_icon="📦", layout="centered")
st.title("📦 Supply Chain SLA Late Delivery Simulator")
st.markdown("Sistem Pendukung Keputusan berbasis Deep Learning PyTorch.")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load('scaler_supply_chain.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    
    layer_sizes = (128, 64)
    input_dim = len(feature_cols)
    
    class DynamicSupplyChainMLP(nn.Module):
        def __init__(self, input_dim, layer_sizes):
            super(DynamicSupplyChainMLP, self).__init__()
            layers = []
            prev_dim = input_dim
            for h_dim in layer_sizes:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.20))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())
            self.network = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.network(x)
            
    model = DynamicSupplyChainMLP(input_dim, layer_sizes)
    model.load_state_dict(torch.load('supply_chain_mlp_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model, scaler, feature_cols

model, scaler, feature_cols = load_artifacts()

st.subheader("📝 Parameter Pesanan")
col1, col2 = st.columns(2)
with col1:
    scheduled_days = st.slider("Target Hari Pengiriman", 0, 4, 4)
    shipping_mode = st.selectbox("Mode Pengiriman", ["Standard Class", "Second Class", "First Class", "Same Day"])
    order_hour = st.slider("Jam Pemesanan", 0, 23, 10)
with col2:
    sales_val = st.number_input("Nilai Penjualan ($)", value=200.0)
    benefit_val = st.number_input("Estimasi Keuntungan ($)", value=30.0)
    item_qty = st.number_input("Kuantitas", min_value=1, max_value=5, value=1)
    product_price = st.number_input("Harga Produk ($)", value=50.0)

market = st.selectbox("Wilayah Pasar", ["Pacific Asia", "Europe", "LATAM", "USCA", "Africa"])
department = st.selectbox("Departemen", ["Fan Apparel", "Golf", "Footwear", "Other"])

if st.button("🚀 Jalankan Prediksi"):
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_cols)
    if 'Days_for_shipment__scheduled_' in input_df.columns: input_df['Days_for_shipment__scheduled_'] = scheduled_days
    if 'Order_Hour' in input_df.columns: input_df['Order_Hour'] = order_hour
    if 'After_Cutoff_12PM' in input_df.columns: input_df['After_Cutoff_12PM'] = 1 if order_hour >= 12 else 0
    if 'Benefit_per_order' in input_df.columns: input_df['Benefit_per_order'] = benefit_val
    if 'Sales_per_customer' in input_df.columns: input_df['Sales_per_customer'] = sales_val
    if 'Order_Item_Quantity' in input_df.columns: input_df['Order_Item_Quantity'] = item_qty
    if 'Product_Price' in input_df.columns: input_df['Product_Price'] = product_price
    if 'Benefit_to_Sales_Ratio' in input_df.columns: input_df['Benefit_to_Sales_Ratio'] = benefit_val / sales_val if sales_val != 0 else 0.0

    for col_name, val in [("Shipping_Mode", shipping_mode), ("Market", market), ("Department_Name", department)]:
        target_col = f"{col_name}_{val.replace(' ', '_')}"
        if target_col in input_df.columns: input_df[target_col] = 1.0

    scaled_input = scaler.transform(input_df)
    with torch.no_grad():
        prob_risk = model(torch.tensor(scaled_input, dtype=torch.float32)).item() * 100.0

    st.divider()
    st.metric(label="Probabilitas Risiko Keterlambatan", value=f"{prob_risk:.2f}%")
    if prob_risk >= 80.0: st.error("🚨 KRITIS: Critical SLA Risk (>80%) — Wajib Prioritas!")
    elif prob_risk >= 60.0: st.warning("⚠️ TINGGI: High Risk (60-80%)")
    elif prob_risk >= 30.0: st.info("ℹ️ SEDANG: Moderate Risk (30-60%)")
    else: st.success("✅ AMAN: Low Risk (<30%)")
