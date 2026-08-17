import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

st.set_page_config(page_title="Bean Classification", layout="wide")

@st.cache_resource
def load_resources():
    scaler = joblib.load('model/scaler.joblib')
    le = joblib.load('model/label_encoder.joblib')
    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "K-Nearest Neighbor": "k-nearest_neighbor.joblib",
        "Naive Bayes (Gaussian)": "naive_bayes_(gaussian).joblib",
        "Random Forest": "random_forest.joblib"
    }
    return scaler, le, model_files

scaler, le, model_files = load_resources()

st.title("🫘 Bean Classification")

st.sidebar.header("Model Selection")
selected_name = st.sidebar.selectbox("Choose a model", list(model_files.keys()))
model = joblib.load(f"model/{model_files[selected_name]}")

st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type="csv")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Test Data Preview")
    st.dataframe(test_df.head())

    if st.button("Evaluate Model on Uploaded Data"):
        X_test = test_df.drop('Class', axis=1)
        y_true = le.transform(test_df['Class'])
        
        # Process inputs
        proc_X = scaler.transform(X_test) if selected_name in ["Logistic Regression", "K-Nearest Neighbor"] else X_test
        
        y_pred = model.predict(proc_X)
        y_proba = model.predict_proba(proc_X)

        # Metrics
        st.subheader("📊 Evaluation Metrics")
        m1, m2, m3 = st.columns(3)
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro', labels=range(len(le.classes_)))
        mcc = matthews_corrcoef(y_true, y_pred)

        m1.metric("Accuracy", f"{acc:.4f}")
        m2.metric("AUC (Macro)", f"{auc:.4f}")
        m3.metric("MCC", f"{mcc:.4f}")
        
        m4, m5, m6 = st.columns(3)
        m4.metric("Precision", f"{precision_score(y_true, y_pred, average='macro'):.4f}")
        m5.metric("Recall", f"{recall_score(y_true, y_pred, average='macro'):.4f}")
        m6.metric("F1 Score", f"{f1_score(y_true, y_pred, average='macro'):.4f}")

        st.subheader("📝 Classification Report")
        report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True, labels=range(len(le.classes_)))
        st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("Please upload a test CSV file (e.g., test_data.csv) in the sidebar to begin evaluation.")
