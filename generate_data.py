import pandas as pd
import random

rows = []

for i in range(5000):

    orders_per_month = random.randint(1, 20)
    avg_order_value = random.randint(100, 1000)
    last_order_days = random.randint(1, 60)
    customer_rating = round(random.uniform(2.5, 5.0), 1)
    discount_used_before = random.randint(0, 1)

    if (
        orders_per_month > 8
        and avg_order_value > 400
        and last_order_days < 7
    ):
        ordered_without_discount = 1
    else:
        ordered_without_discount = random.randint(0, 1)

    recommended_discount = max(
        5,
        min(
            50,
            50 - (orders_per_month * 2) + (last_order_days // 2)
        )
    )

    rows.append(
        {
            'user_id': i,
            'orders_per_month': orders_per_month,
            'avg_order_value': avg_order_value,
            'last_order_days': last_order_days,
            'customer_rating': customer_rating,
            'discount_used_before': discount_used_before,
            'ordered_without_discount': ordered_without_discount,
            'recommended_discount': recommended_discount
        }
    )

df = pd.DataFrame(rows)

df.to_csv('customers.csv', index=False)

print("Dataset Created Successfully")