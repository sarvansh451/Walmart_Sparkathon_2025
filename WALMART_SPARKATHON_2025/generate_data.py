import pandas as pd
import random

queries = [
    "iphone 15 discount", "cheap headphones", "best grocery deal",
    "limited edition shoes", "budget air fryer", "deal of the day"
]
categories = ["electronics", "grocery", "fashion", "appliances"]

data = []
for i in range(500):
    query = random.choice(queries)
    views = random.randint(1, 20)
    time_spent = random.randint(10, 300)
    clicked_coupon = random.choice(["yes", "no"])
    shipping_checked = random.choice(["yes", "no"])
    checkout_pressed = random.choice(["yes", "no"])
    category = random.choice(categories)

    # Fake checkout if tactical
    fake = (
        (time_spent < 90 and shipping_checked == "no" and clicked_coupon == "yes")
        or (checkout_pressed == "yes" and shipping_checked == "no")
    )
    real_checkout = 0 if fake else 1

    data.append([
        f"{i:03}", query, category, views, time_spent,
        clicked_coupon, shipping_checked, checkout_pressed, real_checkout
    ])

df = pd.DataFrame(data, columns=[
    "session_id", "query", "category", "views", "time_spent", 
    "clicked_coupon", "shipping_checked", "checkout_pressed", "real_checkout"
])
df.to_csv("checkout_sessions.csv", index=False)
print("✅ Dataset saved as checkout_sessions.csv")
