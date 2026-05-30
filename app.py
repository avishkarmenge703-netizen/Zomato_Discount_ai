import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="Zomato Discount AI",
    page_icon="🍕",
    layout="wide"
)

# --------------------------------
# TITLE
# --------------------------------

st.title("🍕 Zomato Discount Optimization AI")

st.markdown(
    """
    AI-powered system to optimize discounts,
    improve customer retention,
    and increase profitability.
    """
)

# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

# --------------------------------
# NO FILE
# --------------------------------

if uploaded_file is None:

    st.info(
        "Upload customers.csv to start analysis."
    )

    st.stop()

# --------------------------------
# LOAD DATA
# --------------------------------

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(f"CSV Error: {e}")

    st.stop()

# --------------------------------
# COLUMN VALIDATION
# --------------------------------

required_columns = [
    "orders_per_month",
    "avg_order_value",
    "last_order_days",
    "customer_rating",
    "discount_used_before",
    "ordered_without_discount",
    "recommended_discount"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    st.error(
        f"Missing columns: {missing}"
    )

    st.stop()

# --------------------------------
# TRAIN MODELS
# --------------------------------

@st.cache_resource
def train_models(data):

    X = data[
        [
            "orders_per_month",
            "avg_order_value",
            "last_order_days",
            "customer_rating",
            "discount_used_before"
        ]
    ]

    y_class = data[
        "ordered_without_discount"
    ]

    y_reg = data[
        "recommended_discount"
    ]

    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    regressor = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    classifier.fit(X, y_class)

    regressor.fit(X, y_reg)

    return classifier, regressor


classification_model, regression_model = train_models(
    df
)

# --------------------------------
# KPIs
# --------------------------------

total_customers = len(df)

avg_order = round(
    df["avg_order_value"].mean(),
    2
)

loyal_customers = len(
    df[df["orders_per_month"] > 10]
)

churn_risk = len(
    df[df["last_order_days"] > 20]
)

# --------------------------------
# KPI CARDS
# --------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Customers",
    total_customers
)

c2.metric(
    "Avg Order Value",
    f"₹{avg_order}"
)

c3.metric(
    "Loyal Customers",
    loyal_customers
)

c4.metric(
    "Churn Risk",
    churn_risk
)

st.divider()

# --------------------------------
# TABS
# --------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "AI Prediction",
        "Profit Simulator",
        "Insights",
        "Report"
    ]
)

# --------------------------------
# TAB 1
# --------------------------------

with tab1:

    st.subheader(
        "Customer Intelligence"
    )

    selected_customer = st.selectbox(
        "Select Customer",
        df.index
    )

    customer = df.iloc[
        selected_customer
    ]

    features = np.array(
        [[
            customer["orders_per_month"],
            customer["avg_order_value"],
            customer["last_order_days"],
            customer["customer_rating"],
            customer["discount_used_before"]
        ]]
    )

    prediction = (
        classification_model.predict(
            features
        )[0]
    )

    discount = (
        regression_model.predict(
            features
        )[0]
    )

    if prediction == 1:

        st.success(
            "Customer likely to order WITHOUT discount."
        )

    else:

        st.warning(
            "Customer likely NEEDS discount."
        )

    st.info(
        f"Recommended Discount: {round(discount,2)}%"
    )

    st.write("### Personalized Offer")

    if customer["avg_order_value"] > 700:

        st.success(
            "Premium Dining Coupon"
        )

    elif customer["last_order_days"] > 20:

        st.warning(
            "Win-Back Coupon"
        )

    elif customer["orders_per_month"] > 10:

        st.info(
            "Loyalty Reward"
        )

    else:

        st.write(
            "Standard Offer"
        )

# --------------------------------
# TAB 2
# --------------------------------

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

    traditional_cost = (
        customers * coupon
    )

    optimized_cost = (
        int(customers * 0.3)
        * coupon
    )

    savings = (
        traditional_cost
        - optimized_cost
    )

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Traditional Cost",
        f"₹{traditional_cost:,}"
    )

    p2.metric(
        "AI Cost",
        f"₹{optimized_cost:,}"
    )

    p3.metric(
        "Estimated Savings",
        f"₹{savings:,}"
    )

# --------------------------------
# TAB 3
# --------------------------------

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
        churn_risk,
        total_customers - loyal_customers
    ]

    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

# --------------------------------
# TAB 4
# --------------------------------

with tab4:

    st.subheader(
        "Download Report"
    )

    report = f"""
Zomato Discount Optimization Report

Total Customers: {total_customers}

Average Order Value: ₹{avg_order}

Loyal Customers: {loyal_customers}

Churn Risk Customers: {churn_risk}

Estimated Savings: ₹{savings}
"""

    st.download_button(
        label="Download Report",
        data=report,
        file_name="zomato_report.txt"
    )

# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Built by Avishkar Menge"
)
