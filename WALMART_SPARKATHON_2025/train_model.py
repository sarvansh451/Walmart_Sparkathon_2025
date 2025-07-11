import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv("checkout_sessions.csv")
df["clicked_coupon"] = df["clicked_coupon"].map({"yes": 1, "no": 0})
df["shipping_checked"] = df["shipping_checked"].map({"yes": 1, "no": 0})
df["checkout_pressed"] = df["checkout_pressed"].map({"yes": 1, "no": 0})
df = pd.get_dummies(df, columns=["category"])

X = df.drop(columns=["session_id", "query", "real_checkout"])
y = df["real_checkout"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print("✅ model.pkl saved.")

# Optional: evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
