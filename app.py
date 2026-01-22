import streamlit as st
import numpy as np
import joblib

# Load saved objects
svm_linear = joblib.load("svm_linear.pkl")
svm_poly = joblib.load("svm_poly.pkl")
svm_rbf = joblib.load("svm_rbf.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# App Title
st.title("🏦 Smart Loan Approval System")

st.write("""
This system uses **Support Vector Machines (SVM)** to predict whether
a loan should be **Approved or Rejected** based on applicant details.
""")

# Sidebar Inputs
st.sidebar.header("📋 Applicant Information")

Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Married = st.sidebar.selectbox("Married", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.sidebar.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.sidebar.number_input("Loan Amount Term", min_value=0)
Credit_History = st.sidebar.radio("Credit History", ["Yes", "No"])

Credit_History = 1 if Credit_History == "Yes" else 0

# Encode categorical values using SAME encoders
input_data = {
    "Gender": Gender,
    "Married": Married,
    "Dependents": Dependents,
    "Education": Education,
    "Self_Employed": Self_Employed,
    "Property_Area": Property_Area,
}

encoded_features = []
for col, val in input_data.items():
    encoded_features.append(label_encoders[col].transform([val])[0])

final_input = np.array([[
    *encoded_features,
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History
]])

# Feature scaling
final_input_scaled = scaler.transform(final_input)

# Kernel selection
st.subheader("🧠 Select SVM Kernel")
kernel = st.radio("Choose Kernel", ["Linear SVM", "Polynomial SVM", "RBF SVM"])

# Prediction
if st.button("🔍 Check Loan Eligibility"):

    if kernel == "Linear SVM":
        model = svm_linear
        kernel_used = "Linear"
    elif kernel == "Polynomial SVM":
        model = svm_poly
        kernel_used = "Polynomial"
    else:
        model = svm_rbf
        kernel_used = "RBF"

    prediction = model.predict(final_input_scaled)[0]
    confidence = model.decision_function(final_input_scaled)[0]

    st.subheader("📊 Loan Decision")

    if prediction == 1:
        st.success("✅ Loan Approved")
        decision_text = "likely"
    else:
        st.error("❌ Loan Rejected")
        decision_text = "unlikely"

    st.write(f"**Kernel Used:** {kernel_used}")
    st.write(f"**Confidence Score:** {round(confidence, 2)}")

    st.info(
        f"""
        **Business Explanation:**  
        Based on **credit history, income stability, and employment profile**,  
        the applicant is **{decision_text}** to repay the loan.
        """
    )
