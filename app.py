import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction System",
    page_icon="🩺",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #4338ca 0%, #2563eb 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load and Train Model (Cached for performance)
@st.cache_resource
def load_and_train_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=column_names)
    
    X = df[['Age', 'Glucose', 'BMI']]
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    return model, accuracy, cv_scores, df

model, test_accuracy, cv_scores, df_data = load_and_train_model()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", ["🏠 Home & Predictor", "📊 Model Performance"])

# Function to generate PDF Report
def generate_pdf(age, glucose, bmi, result_text):
    pdf_path = "medical_report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, "Medical Risk Assessment Report")
    c.drawString(100, 720, "--------------------------------------------------------")
    c.drawString(100, 690, f"Patient Age: {age}")
    c.drawString(100, 660, f"Glucose Level: {glucose} mg/dL")
    c.drawString(100, 630, f"BMI (Body Mass Index): {bmi}")
    c.drawString(100, 600, f"Assessment Result: {result_text}")
    c.drawString(100, 550, "Note: This is an AI-generated screening report. Please consult a doctor.")
    c.save()
    return pdf_path

# Home Page: Interactive Predictor
if page == "🏠 Home & Predictor":
    st.title("🩺 Diabetes Risk Prediction System")
    st.markdown("### Enter patient details below to get an instant AI-powered health assessment and visual breakdown.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
    with col2:
        glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
    with col3:
        bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0)
        
    if st.button("Predict Risk"):
        new_patient = pd.DataFrame([[age, glucose, bmi]], columns=['Age', 'Glucose', 'BMI'])
        prediction = model.predict(new_patient)
        
        st.divider()
        st.subheader("Prediction Result:")
        
        if prediction[0] == 1:
            res_text = "At Risk of Diabetes"
            st.error(f"⚠️ The model predicts that the patient is **{res_text}**. (Please consult a doctor).")
        else:
            res_text = "Healthy"
            st.success(f"✅ The model predicts that the patient is **{res_text}**.")
        
        # Visual Comparison Charts
        st.subheader("📊 Visual Comparison with Dataset Averages")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        
        # Glucose Comparison
        ax[0].bar(['Your Glucose', 'Dataset Avg'], [glucose, df_data['Glucose'].mean()], color=['#4f46e5', '#cbd5e1'])
        ax[0].set_title("Glucose Level Comparison")
        ax[0].set_ylabel("mg/dL")
        
        # BMI Comparison
        ax[1].bar(['Your BMI', 'Dataset Avg'], [bmi, df_data['BMI'].mean()], color=['#3b82f6', '#cbd5e1'])
        ax[1].set_title("BMI Comparison")
        
        st.pyplot(fig)
        
        # PDF Generation Section (Directly outside form, works instantly!)
        pdf_file = generate_pdf(age, glucose, bmi, res_text)
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="📥 Download Medical Report (PDF)",
                data=f,
                file_name="Medical_Report.pdf",
                mime="application/pdf"
            )

# Performance Page
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Metrics")
    st.write("Evaluating the Random Forest classifier trained on the PIMA Indians Diabetes dataset:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Test Accuracy", value=f"{test_accuracy * 100:.2f}%")
    with col2:
        st.metric(label="Cross-Validation Accuracy", value=f"{cv_scores.mean() * 100:.2f}%")
        
    st.info("The model uses core health metrics (Age, Glucose, and BMI) to deliver real-time classifications and visualizations.")