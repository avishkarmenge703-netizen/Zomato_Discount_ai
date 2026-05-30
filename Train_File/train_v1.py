import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('customers.csv')

X = df[
    [
        'orders_per_month',
        'avg_order_value',
        'last_order_days',
        'customer_rating',
        'discount_used_before'
    ]
]

y = df['ordered_without_discount']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

joblib.dump(model, 'model_v1.pkl')

print("Model Saved")
