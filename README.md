# 🛒 Walmart Checkout Intent Analyzer – Sparkathon 2025

## 🚀 Overview

This project is built for **Walmart Sparkathon 2025** and tackles the challenge of **smart inventory management and coupon optimization**. By analyzing real-time customer behavior data during online shopping sessions, the tool predicts whether a user is genuinely intending to checkout or exhibiting tactical behavior (e.g., exploiting coupons).

The app provides:
- ML-based checkout seriousness predictions
- Coupon cost optimization
- Session analysis & visual insights
- A Streamlit-powered analytics dashboard
- Groq LLM-based explanation engine

---

## 🧠 Problem Statement

Retailers face challenges like:
- Misused coupons by non-serious users
- High marketing costs
- Difficulty predicting user intent in real-time

Our solution uses **machine learning and behavioral data** to detect:
- 🎯 Real customers likely to buy
- 🎭 Tactical users mimicking genuine behavior
- 💸 Opportunities to save costs via smart coupon strategy

---

## ✅ Key Features

| Module | Description |
|--------|-------------|
| 🔍 Analyze Session | Predicts seriousness & camouflage score of user |
| 📊 Analytics Dashboard | Visualizes session data using charts and heatmaps |
| 🧠 LLM Reasoning (Groq) | Explains predictions with natural language |
| 💸 Coupon Cost Optimization | Projects savings via smarter coupon control |
| 🧾 Upload CSV | Upload or use default session data for batch analysis |
| 📥 Export Report | Download processed report as CSV |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML Model:** Scikit-learn (Logistic Regression or similar)
- **Visualization:** Plotly
- **LLM:** Groq API (LLaMA 3)
- **Other libs:** Pandas, NumPy, Joblib

---

## 🧪 Model Input Features

```text
views, time_spent, clicked_coupon, shipping_checked, checkout_pressed,
category_appliances, category_electronics, category_fashion, category_grocery
