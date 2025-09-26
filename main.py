import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from datetime import datetime as dt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# =====================
# Folder Setup (Use existing folders)
# =====================
EDA_DIR = "EDA Results"
MODELS_DIR = "Models"
RESULTS_DIR = "Model Results"
os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================
# Load Dataset (UCI Breast Cancer Diagnostic)
# =====================
print("Loading Dataset (UCI Breast Cancer Diagnostic)..........")
from ucimlrepo import fetch_ucirepo
breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features
y = breast_cancer.data.targets

# Rename target column for clarity
y = y.squeeze()                # flatten to Series
y = y.map({"B": 0, "M": 1})    # convert labels to numeric

# =====================
# Remove Highly Correlated Features
# =====================
print("Removing highly correlated features..........")
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
print(f"Dropped {len(high_corr_features)} highly correlated features: {high_corr_features}")
X = X.drop(columns=high_corr_features)

# =====================
# EDA
# =====================
print('Starting EDA.........')
plt.figure(figsize=(15,10))
plt.title("Box Plots for Outliers")
X.boxplot()
plt.xticks(rotation=90)
plt.savefig(os.path.join(EDA_DIR, f"boxplot_{dt.now().strftime('%y_%b_%d_%H_%M')}.jpg"))
plt.close()

plt.figure(figsize=(10,10))
sns.heatmap(X.corr(), annot=False, cmap='Blues')
plt.savefig(os.path.join(EDA_DIR, f"heatmap_{dt.now().strftime('%y_%b_%d_%H_%M')}.jpg"))
plt.close()

# Pairplot
# Pairplot
print("Creating Pairplot..........")
pairplot_df = pd.concat([X, y], axis=1).copy()
pairplot_df["Diagnosis_Label"] = pairplot_df["Diagnosis"].map({0: "Benign", 1: "Malignant"})

sns.pairplot(pairplot_df, hue="Diagnosis_Label", diag_kind="kde")
plt.savefig(os.path.join(EDA_DIR, f"pairplot_{dt.now().strftime('%y_%b_%d_%H_%M')}.jpg"))
plt.close()


# =====================
# Feature Scaling
# =====================
print('Starting Scaling..........')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# =====================
# Train-Test Split (Binary Classification)
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =====================
# Models & Hyperparameters
# =====================
print('Model Training Started.....')
models = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000),
        {"C": [0.01, 0.1, 1, 5, 10, 50],
         "solver": ["newton-cg", "lbfgs", "saga"],
         "multi_class": ["ovr", "multinomial"]}
    ),
    "SVM": (
        SVC(probability=True),
        {"C": [0.01, 0.1, 1, 10, 50],
         "kernel": ["linear", "rbf", "poly", "sigmoid"],
         "gamma": ["scale", "auto"]}
    ),
    "RandomForest": (
        RandomForestClassifier(),
        {"n_estimators": [100, 200, 500],
         "max_depth": [3, 5, 10, None],
         "min_samples_split": [2, 5, 10],
         "min_samples_leaf": [1, 2, 4],
         "max_features": ["sqrt", "log2", None]}
    ),
    "XGBoost": (
        XGBClassifier(eval_metric='logloss', use_label_encoder=False),
        {"n_estimators": [100, 200, 500],
         "max_depth": [3, 5, 7, 10],
         "learning_rate": [0.01, 0.05, 0.1, 0.2],
         "subsample": [0.6, 0.8, 1.0],
         "colsample_bytree": [0.6, 0.8, 1.0]}
    )
}

results = []

# =====================
# Training Function
# =====================
def train_and_evaluate(X_train, X_test, y_train, y_test):
    for name, (model, params) in models.items():
        grid = GridSearchCV(
            model,
            params,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy",
            n_jobs=-1
        )
        grid.fit(X_train, y_train.values.ravel())  # flatten y
        best_model = grid.best_estimator_

        filename = f"{name}_binary.pkl"
        with open(os.path.join(MODELS_DIR, filename), "wb") as f:
            pickle.dump(best_model, f)

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        y_train_prob = best_model.predict_proba(X_train)[:, 1]
        y_test_prob = best_model.predict_proba(X_test)[:, 1]
        auc_train = roc_auc_score(y_train, y_train_prob)
        auc_test = roc_auc_score(y_test, y_test_prob)

        for split, y_true, y_pred, auc in [
            ("Train", y_train, y_train_pred, auc_train),
            ("Test", y_test, y_test_pred, auc_test)
        ]:
            results.append({
                "Model": name,
                "Task": "binary",
                "Split": split,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision_weighted": precision_score(y_true, y_pred, average="weighted"),
                "Recall_weighted": recall_score(y_true, y_pred, average="weighted"),
                "F1_weighted": f1_score(y_true, y_pred, average="weighted"),
                "Precision_macro": precision_score(y_true, y_pred, average="macro"),
                "Recall_macro": recall_score(y_true, y_pred, average="macro"),
                "F1_macro": f1_score(y_true, y_pred, average="macro"),
                "AUC ROC": auc
            })

# =====================
# Train Models
# =====================
train_and_evaluate(X_train, X_test, y_train, y_test)

# =====================
# Save Results
# =====================
results_df = pd.DataFrame(results)
results_file = os.path.join(RESULTS_DIR, f"model_results_{dt.now().strftime('%y_%b_%d_%H_%M')}.xlsx")
results_df.to_excel(results_file, index=False)

print(" Pipeline completed! All models, plots, and results are saved.")

