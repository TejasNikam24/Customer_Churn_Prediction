# -----------------------------
# 🔐 ADD YOUR HUGGINGFACE TOKEN HERE
# -----------------------------
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] ="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"

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import streamlit as st
import pandas as pd
import joblib

from langchain_huggingface import HuggingFaceEndpoint

# -----------------------------
# LOAD MODEL & COLUMNS
# -----------------------------
model = joblib.load("churn_model.pkl")
columns = joblib.load("columns.pkl")

# -----------------------------
# LOAD FREE HUGGINGFACE MODEL
# -----------------------------
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    task="text-generation"
)

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Prediction", "Chatbot"])

# =============================
# 🔮 PREDICTION PAGE
# =============================
if page == "Prediction":

    st.title("Customer Churn Prediction")

    st.write("Enter customer details:")

    # Inputs
    tenure = st.slider("Tenure (Months)", 0, 72)
    monthly = st.number_input("Monthly Charges", value=50.0)
    total = st.number_input("Total Charges", value=1000.0)

    if st.button("Predict"):

        # Create full input with all columns
        input_data = pd.DataFrame(columns=columns)
        input_data.loc[0] = 0

        # Fill required values
        input_data["tenure"] = tenure
        input_data["MonthlyCharges"] = monthly
        input_data["TotalCharges"] = total

        # Prediction
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Customer will churn")
        else:
            st.success("✅ Customer will stay")

# =============================
# 🤖 CHATBOT PAGE
# =============================
elif page == "Chatbot":

    st.title("AI Business Assistant")

    st.write("Ask questions about customer churn")

    question = st.text_input("Your Question:")

    if st.button("Ask AI"):

        if question.strip() != "":

            # Prompt Engineering
            prompt = f"""
            You are a telecom business analyst.

            Answer in simple, clear, and professional language.

            Question: {question}
            """

            response = llm.invoke(prompt)

            st.write("💡 Answer:")
            st.write(response)

        else:
            st.warning("Please enter a question.")
