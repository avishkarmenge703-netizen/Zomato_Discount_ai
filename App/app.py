import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import os

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Zomato Discount AI",
    page_icon="🍕",
    layout="wide"
)

# =====================================
# PATHS
# =====================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "Data" / "customers.csv"

MODEL_V1_PATH = ROOT / "Models" / "model_v1.pkl"
MODEL_V2_PATH = ROOT / "Models" / "model_v2.pkl"

# =====================================
# DEBUG SECTION
# =====================================

with st.sidebar:
    st.subheader("System Check")

    st.write("CSV Found:", DATA_PATH.exists())
    st.write("Model V1 Found:", MODEL_V1_PATH.exists())
    st.write("Model V2 Found:", MODEL_V2_PATH.exists())

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_models():

    model1 = joblib.load(MODEL_V1_PATH)
    model2 = joblib.load(MODEL_V2_PATH)

    return model1, model2

# =====================================
# FILE LOADING
# =====================================

try:

    df = load_data()

    model_v1, model_v2 = load_models()

except Exception as e:

    st.error("Application Startup Error")

    st.exception(e)

    st.stop()

# =====================================
# TITLE
# =====================================

st.title("🍕 Zomato Discount Optimization AI")

st.markdown(
    """
    AI-powered system to optimize discounts,
    improve customer retention,
    and increase profitability.
    """
)

# =====================================
# KPI SECTION
# =====================================

total_customers = len(df)

avg_order_value = round(
    df["avg_order_value"].mean(),
    2
)

loyal_customers = len(
    df[df["orders_per_month"] > 10]
)

churn_risk_customers = len(
    df[df["last_order_days"] > 20]
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Customers",
    total_customers
)

col2.metric(
    "Avg Order Value",
    f"₹{avg_order_value}"
)

col3.metric(
    "Loyal Customers",
    loyal_customers
)

col4.metric(
    "Churn Risk",
    churn_risk_customers
)

st.divider()

# =====================================
# TABS
# =====================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Prediction",
        "Profit Simulation",
        "Insights",
        "Report"
    ]
)

# =====================================
# TAB 1
# =====================================

with tab1:

    st.subheader(
        "Customer Discount Prediction"
    )

    customer_index = st.selectbox(
        "Select Customer",
        df.index
    )

    customer = df.iloc[customer_index]

    st.write(customer)

    features = np.array(
        [[
            customer["orders_per_month"],
            customer["avg_order_value"],
            customer["last_order_days"],
            customer["customer_rating"],
            customer["discount_used_before"]
        ]]
    )

    try:

        prediction = model_v1.predict(
            features
        )[0]

        recommended_discount = model_v2.predict(
            features
        )[0]

        if prediction == 1:

            st.success(
                "Customer likely to order without discount."
            )

        else:

            st.warning(
                "Customer likely needs a discount."
            )

        st.info(
            f"Recommended Discount: {round(recommended_discount,2)}%"
        )

    except Exception as e:

        st.error("Prediction Error")

        st.exception(e)

# =====================================
# TAB 2
# =====================================

with tab2:

    st.subheader(
        "Profit Simulation"
    )

    customers = st.slider(
        "Monthly Customers",
        100,
        50000,
        5000
    )

    coupon = st.slider(
        "Average Coupon Value",
        10,
        500,
        100
    )

    traditional_cost = customers * coupon

    ai_cost = int(customers * 0.30) * coupon

    savings = traditional_cost - ai_cost

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Traditional Cost",
        f"₹{traditional_cost:,}"
    )

    c2.metric(
        "AI Optimized Cost",
        f"₹{ai_cost:,}"
    )

    c3.metric(
        "Savings",
        f"₹{savings:,}"
    )

# =====================================
# TAB 3
# =====================================

with tab3:

    st.subheader(
        "Customer Segmentation"
    )

    fig, ax = plt.subplots()

    labels = [
        "Loyal",
        "Churn Risk",
        "Regular"
    ]

    values = [
        loyal_customers,
        churn_risk_customers,
        total_customers - loyal_customers
    ]

    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

# =====================================
# TAB 4
# =====================================

with tab4:

    st.subheader(
        "Business Report"
    )

    report = f"""
Zomato Discount AI Report

Total Customers: {total_customers}

Average Order Value: ₹{avg_order_value}

Loyal Customers: {loyal_customers}

Churn Risk Customers: {churn_risk_customers}

Estimated Savings: ₹{savings}
"""

    st.download_button(
        label="Download Report",
        data=report,
        file_name="zomato_discount_report.txt"
    )

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "Built by Avishkar Menge | Data Analyst & ML Engineer"
)
