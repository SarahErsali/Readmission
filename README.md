# 🏥 Hospital Readmission Predictive Models for Diabetes

## 📊 Introduction

The dataset spans a decade (1999–2008) of clinical care across 130 U.S. hospitals and integrated delivery networks (obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/)).  
Each record reflects a patient diagnosed with diabetes, including details about their hospital stay (up to 14 days), lab tests, and medications.

Inadequate diabetes management not only escalates hospital operational costs due to frequent readmissions but also worsens patient outcomes by increasing the risk of diabetes-related complications, morbidity, and mortality.

---

## 🎯 Business Objective

The goal of this project is to identify and understand the key factors contributing to early hospital readmissions among diabetic patients.

Although various treatment strategies have proven effective in improving clinical outcomes for diabetes, a significant portion of patients either do not receive these treatments or fail to adhere to them after discharge.

This gap in diabetes care leads to costly hospital readmissions and puts patients at increased risk of serious health complications.

By analyzing historical patient data, we aim to build predictive models that can flag individuals at high risk of readmission — supporting proactive care planning and ultimately improving both health outcomes and hospital resource efficiency.

---

## 🤖 Model Development and Evaluation

In this project, several machine learning models were developed and evaluated, including:

- **XGBoost**
- **LightGBM**
- **Random Forest**
- **Ensemble Model** (combining XGBoost and Random Forest)

Each model was implemented in two configurations:

- **Default settings**
- **Tuned using hyperparameter optimization**

Model performance was assessed using key metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **AUC**

To visualize performance, ROC curves and grouped bar charts were included.

To further evaluate predictive reliability, five patient records were used to test each model's ability to forecast readmission:

- **XGBoost**, **Random Forest**, and the **Ensemble model** each has the KPIs of 0.62.
- **LightGBM** KPIs are 0.63.

This result highlights the superior reliability of LightGBM model in clinical risk prediction.

---

## 🚀 Final Product & Dashboard

The final product is an **interactive Dash/Plotly dashboard** designed to deliver insights to both technical and non-technical stakeholders.

### Dashboard Highlights:

- 📌 Single-tab interface for clean, user-friendly exploration.
- 📈 Visual performance comparison across models using:
  - ROC curves
  - KPI bar charts
- 🧠 Prediction output table showcasing how each selected model predicts the readmission outcome for five patient records.
- 🧩 Interactive model selection — view one or multiple models at a time to compare outcomes side-by-side.

This dashboard is a complete end-to-end solution for analyzing model performance and presenting results to clinicians, data scientists, or business stakeholders.
