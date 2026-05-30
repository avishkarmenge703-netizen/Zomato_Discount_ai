# Zomato_Discount_AI

## AI-Powered Discount Optimization System for Food Delivery Platforms

An end-to-end Machine Learning project designed to help food delivery companies optimize discount spending, improve customer retention, and increase profitability through intelligent coupon allocation.

---

## Project Overview

Food delivery companies spend significant amounts on discounts and promotional campaigns. Many discounts are offered to customers who would place orders even without incentives, leading to unnecessary revenue leakage.

This project uses Machine Learning to identify:

* Customers who need discounts
* Customers who will order without discounts
* Recommended discount percentages
* Churn-risk customers
* High-value customers
* Personalized coupon strategies

The system also provides an interactive Streamlit dashboard for business stakeholders.

---

## Business Problem

Traditional discount systems often provide the same offers to all customers.

### Challenges

* Excessive coupon spending
* Low marketing efficiency
* Customer churn
* Reduced profitability
* Lack of personalization

### Solution

An AI-powered system that predicts customer behavior and recommends optimized discount strategies.

---

## Features

### Version 1 — Discount Prediction

Predict whether a customer requires a discount to place an order.

**Machine Learning Task:**
Classification

**Target Variable:**

```text
ordered_without_discount
```

---

### Version 2 — Discount Amount Prediction

Predict the optimal discount percentage for each customer.

**Machine Learning Task:**
Regression

**Target Variable:**

```text
recommended_discount
```

---

### Version 3 — Personalized Coupon Engine

Generate intelligent coupon recommendations based on customer behavior.

Examples:

* Premium Dining Offer
* Loyalty Rewards
* Win-Back Coupons
* Standard Discounts

---

### Version 4 — Business Intelligence Dashboard

Interactive Streamlit dashboard including:

* Customer analytics
* AI predictions
* Profit simulation
* Customer segmentation
* Downloadable business reports

---

## Technology Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Random Forest Classifier
* Random Forest Regressor

### Dashboard

* Streamlit

### Visualization

* Matplotlib

### Model Storage

* Joblib

---

## Project Structure

```text
zomato_discount_ai/
│
├── app.py
├── generate_data.py
├── train_v1.py
├── train_v2.py
├── customers.csv
├── model_v1.pkl
├── model_v2.pkl
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Zomato_Discount_AI.git
```

Navigate to project directory:

```bash
cd Zomato_Discount_AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Generate Dataset

```bash
python generate_data.py
```

This creates:

```text
customers.csv
```

---

## Train Classification Model

```bash
python train_v1.py
```

Output:

```text
model_v1.pkl
```

---

## Train Regression Model

```bash
python train_v2.py
```

Output:

```text
model_v2.pkl
```

---

## Run Streamlit Dashboard

```bash
streamlit run app.py
```

---

## Dashboard Capabilities

### Business KPIs

* Total Customers
* Average Order Value
* Loyal Customers
* Churn Risk Customers

### AI Predictions

* Discount Required Prediction
* Recommended Discount Percentage
* Personalized Offer Generation

### Profitability Simulator

Compare:

* Traditional Discount Strategy
* AI-Based Discount Strategy

Estimate monthly savings.

### Customer Segmentation

Identify:

* High Value Customers
* Loyal Customers
* Churn Risk Customers

---

## Business Impact

This solution helps food delivery companies:

* Reduce unnecessary coupon spending
* Improve marketing efficiency
* Increase profitability
* Improve customer retention
* Personalize user experience
* Enable data-driven decision making

---

## Future Improvements

### Machine Learning

* XGBoost
* LightGBM
* CatBoost

### Deep Learning

* TensorFlow
* PyTorch

### Real-Time Data

* Weather APIs
* Traffic APIs
* Location Intelligence

### Deployment

* AWS
* Docker
* Kubernetes

### Database Integration

* PostgreSQL
* MongoDB

---

## Sample Use Cases

* Food Delivery Platforms
* Quick Commerce Companies
* E-commerce Coupon Optimization
* Customer Retention Systems
* Loyalty Programs

App Link: https://zomatodiscountai-yhusdyrytl3lyzgygclxtn.streamlit.app/
---

## Author

### Avishkar Menge

Data Analyst | Aspiring Machine Learning Engineer

Focused on building AI systems that solve real-world business problems through data-driven decision making.

---

## License

This project is intended for educational, portfolio, and demonstration purposes.
