import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Page Title and Subtitle
st.title("🎯 Customer Churn Predictor")
st.write("Enter the customer's data below to see if they are at risk of leaving.")

# 2. Creating Input Boxes for the User
st.subheader("Customer Behavior Data")

last_purchase = st.slider("Days since last purchase", min_value=1, max_value=60, value=15)
total_purchases = st.slider("Total months with company", min_value=1, max_value=72, value=12)
total_spend = st.number_input("Monthly bill ($)", min_value=10.0, max_value=150.0, value=70.0)

# 3. Simple scoring logic (Simulating our RFM math from earlier)
r_score = 4 if last_purchase <= 90 else (3 if last_purchase <= 180 else 2)
f_score = 4 if total_purchases >= 30 else (3 if total_purchases >= 15 else 2)
m_score = 4 if total_spend >= 1000 else (3 if total_spend >= 500 else 2)

import joblib

# Load your real trained model
model = joblib.load('churn_model.pkl')

# 5. The Prediction Button
if st.button("Predict Churn Risk"):
    # Put the user's inputs into a list
    user_data = pd.DataFrame(
    [[last_purchase, total_purchases, total_spend, r_score, f_score, m_score]],
    columns=['LastPurchaseDaysAgo', 'TotalPurchases', 'TotalSpend', 'R', 'F', 'M']
)
    
    # Make the prediction and get the probability percentage
    churn_probability = model.predict_proba(user_data)[0][1]

    st.markdown("---")
    # If the model thinks there is more than a 30% chance they will leave
    if churn_probability > 0.30:
        st.error(f"🚨 ALERT: Churn Risk is {churn_probability*100:.1f}%! High risk.")
        st.write("Strategy: Send them a retention discount immediately.")
    else:
        st.success(f"✅ Good News: Churn Risk is only {churn_probability*100:.1f}%. Likely to stay.")
        st.write("Strategy: Keep them engaged with normal loyalty rewards.")