import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Page Config
st.set_page_config(
    page_title="AI Discount Optimizer",
    layout="wide"
)

# Load Models (Fix 1 + Fix 2)
base_path = os.path.dirname(__file__)  # current file path
classification_model = joblib.load(os.path.join(base_path, "../Models/model_v1.pkl"))
regression_model = joblib.load(os.path.join(base_path, "../Models/model_v2.pkl"))

# Sidebar
st.sidebar.title("AI Discount Optimizer")

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer Dataset",
    type=["csv"]
)

# Main Title
st.title("AI-Powered Discount Optimization System")
st.write("Optimize discounts, reduce coupon wastage, and improve profitability.")

# Load Dataset
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # KPIs
    total_customers = len(df)
    avg_order = round(df['avg_order_value'].mean(), 2)
    loyal_customers = len(df[df['orders_per_month'] > 10])
    churn_risk = len(df[df['last_order_days'] > 20])

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", total_customers)
    col2.metric("Average Order Value", f"₹{avg_order}")
    col3.metric("Loyal Customers", loyal_customers)
    col4.metric("Churn Risk Users", churn_risk)

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["AI Predictions", "Profit Simulation", "Customer Insights", "Download Report"]
    )

    # TAB 1
    # TAB 1

with tab1:
    st.subheader("Customer Discount Intelligence")

    # Show customer IDs instead of raw index
    if "customer_id" in df.columns:
        customer_id = st.selectbox("Select Customer", df["customer_id"])
        customer = df[df["customer_id"] == customer_id].iloc[0]
    else:
        # fallback: use row number
        customer_index = st.selectbox("Select Customer Row", range(len(df)))
        customer = df.iloc[customer_index]

    input_data = np.array([[
        customer.get('orders_per_month', 0),
        customer.get('avg_order_value', 0),
        customer.get('last_order_days', 0),
        customer.get('customer_rating', 0),
        customer.get('discount_used_before', 0)
    ]])

    prediction = classification_model.predict(input_data)[0]
    discount = regression_model.predict(input_data)[0]

    st.write("### AI Recommendation")

    if prediction == 1:
        st.success("Customer likely to order WITHOUT discount.")
    else:
        st.error("Customer likely NEEDS discount.")

    st.info(f"Recommended Discount: {round(discount, 2)}%")



    # TAB 2
    with tab2:
        st.subheader("Profitability Simulation")
        customers = st.slider("Monthly Customers", 100, 50000, 5000)
        discount_amount = st.slider("Average Coupon Amount", 10, 500, 100)

        traditional_cost = customers * discount_amount
        optimized_customers = int(customers * 0.3)
        optimized_cost = optimized_customers * discount_amount
        savings = traditional_cost - optimized_cost

        col1, col2, col3 = st.columns(3)
        col1.metric("Traditional Cost", f"₹{traditional_cost}")
        col2.metric("AI Optimized Cost", f"₹{optimized_cost}")
        col3.metric("Estimated Savings", f"₹{savings}")

    # TAB 3
    with tab3:
        st.subheader("Customer Intelligence")
        fig, ax = plt.subplots()
        segments = [loyal_customers, churn_risk, total_customers - loyal_customers]
        labels = ["Loyal", "Churn Risk", "Regular"]
        ax.pie(segments, labels=labels, autopct='%1.1f%%')
        st.pyplot(fig)

    # TAB 4
    with tab4:
        st.subheader("Download Business Report")
        report = f'''
AI Discount Optimization Report

Total Customers: {total_customers}
Average Order Value: ₹{avg_order}
Loyal Customers: {loyal_customers}
Churn Risk Customers: {churn_risk}
Estimated Savings: ₹{savings}
'''
        st.download_button(
            label="Download Report",
            data=report,
            file_name="business_report.txt"
        )

else:
    st.info("Please upload a customer dataset to continue.")
