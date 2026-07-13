# Telco Customer Churn Prediction - End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6-orange?style=flat&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

> **DevelopersHub Corporation - AI/ML Engineering Internship**

A production-ready, reusable machine learning pipeline for predicting customer churn in the telecommunications industry. Built using the Scikit-learn Pipeline API with full preprocessing, model training, hyperparameter tuning, and joblib export.

---

## Overview

Customer churn is one of the most critical challenges in the telecommunications industry. Retaining an existing customer costs significantly less than acquiring a new one. This project builds a clean, modular ML pipeline that predicts whether a customer will leave the service - enabling proactive retention strategies.

The pipeline encapsulates the entire ML workflow as a single exportable object: from raw data ingestion to a deployable prediction artifact.

---

## Features

- Full scikit-learn `Pipeline` API implementation - preprocessing and modeling in one object
- Separate sub-pipelines for numerical (impute + scale) and categorical (impute + one-hot encode) features via `ColumnTransformer`
- Two classifiers: **Logistic Regression** and **Random Forest**
- Hyperparameter tuning with **GridSearchCV** and **StratifiedKFold** cross-validation
- Evaluation with **Accuracy**, **F1 Score**, and **ROC-AUC**
- Visualizations: class distribution, confusion matrix, ROC curves, feature importance
- Pipeline export with **joblib** - load and predict without retraining
- Prevents data leakage by fitting transformers only on training data

---

## Results

| Model | Accuracy | F1 (Churn) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~80% | ~0.60 | ~0.84 |
| Random Forest | ~79% | ~0.58 | ~0.83 |

> Exact values printed after running the notebook on the dataset.

**Key finding:** Month-to-month contract customers churn at significantly higher rates than those on 1 or 2-year contracts. Tenure and monthly charges are the strongest numerical predictors.

---

## Project Structure

```
telco-churn-ml-pipeline/
├── telco_churn_pipeline.ipynb     # Main notebook - full pipeline
├── churn_pipeline.pkl             # Exported pipeline (generated after running)
├── requirements.txt               # Dependencies
├── README.md
└── visualizations/
    ├── churn_distribution.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    └── feature_importance.png
```

---

## Dataset

**Telco Customer Churn** dataset by IBM - available on Kaggle:
[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it in the project root before running the notebook.

---

## Setup and Installation

**1. Clone the repository**
```bash
git clone https://github.com/RabiyaMalik242/telco-churn-ml-pipeline.git
cd telco-churn-ml-pipeline
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the root folder.

**4. Run the notebook**
```bash
jupyter notebook telco_churn_pipeline.ipynb
```

---

## Usage - Load the Exported Pipeline

After running the notebook, the trained pipeline is saved as `churn_pipeline.pkl`. Load it anywhere:

```python
import joblib
import pandas as pd

# Load pipeline
pipeline = joblib.load('churn_pipeline.pkl')

# Predict on new raw data (no preprocessing needed)
new_customer = pd.DataFrame([{
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes',
    'Dependents': 'No', 'tenure': 12, 'PhoneService': 'Yes',
    'MultipleLines': 'No', 'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No',
    'StreamingTV': 'No', 'StreamingMovies': 'No',
    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 70.35, 'TotalCharges': 844.2
}])

prediction  = pipeline.predict(new_customer)
probability = pipeline.predict_proba(new_customer)[:, 1]

print(f'Churn Prediction : {"Yes" if prediction[0] == 1 else "No"}')
print(f'Churn Probability: {probability[0]:.2%}')
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | Scikit-learn |
| Pipeline API | `sklearn.pipeline.Pipeline` |
| Preprocessing | `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer` |
| Models | `LogisticRegression`, `RandomForestClassifier` |
| Tuning | `GridSearchCV`, `StratifiedKFold` |
| Export | `joblib` |
| Visualization | Matplotlib, Seaborn |
| Data | Pandas, NumPy |

---

## Skills Demonstrated

- End-to-end ML pipeline construction with scikit-learn
- Production-readiness practices (pipeline export, data leakage prevention)
- Hyperparameter tuning and cross-validation
- Binary classification evaluation (Accuracy, F1, ROC-AUC)
- Handling class imbalance with `class_weight='balanced'`
- Model comparison and selection

---

## Author

**Rabiya Malik** - 
BS Software Engineering

AI/ML Engineering Intern @ DevelopersHub Corporation

[![GitHub](https://img.shields.io/badge/GitHub-RabiyaMalik242-black?style=flat&logo=github)](https://github.com/RabiyaMalik242)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/rabiya-malik-325848336/)

---

## License

This project is licensed under the MIT License.
