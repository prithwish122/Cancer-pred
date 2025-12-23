"""
XAI ByteA - Feature Engineering & Model Training (Optimized for Accuracy)

Rebuilt clean pipeline after reset:
- Robust preprocessing (encoding, imputation, scaling)
- Single strong model (RandomForest) with focused hyperparameter search
- Optimizes for accuracy via cross-validation (no SMOTE / imblearn dependency)
"""

import time
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.ensemble import RandomForestClassifier


# --------------- helpers ---------------
def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S]"), msg)
    sys.stdout.flush()



t0 = time.time()
log("Start")

# ---------- CONFIG ----------
FAST_MODE = True  # Set to False to allow deeper search (slower but possibly better accuracy)
# Use F1 so we jointly optimize precision & recall, which usually improves both recall and overall accuracy
PRIMARY_METRIC = "f1"


# ---------- Load data ----------
log("Loading dataset...")
df = pd.read_excel("AI_ByteA_CleanedDataset.xlsx", engine="openpyxl")
log(f"Data shape: {df.shape}")


# ---------- Encode categoricals ----------
log("Encoding categoricals...")
label_encoders = {}
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    if col != "PatientID":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le


# ---------- Target & features ----------
target_col = "Diagnosis" if "Diagnosis" in df.columns else df.columns[-1]
drop_cols = ["PatientID", target_col] if "PatientID" in df.columns else [target_col]

X = df.drop(columns=drop_cols, errors="ignore")
y = df[target_col].astype(int)


# ---------- Impute missing ----------
log("Imputing missing values with KNNImputer...")
imputer = KNNImputer(n_neighbors=3 if FAST_MODE else 5)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


# ---------- Scale (no polynomial expansion) ----------
log("Scaling features with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
log(f"Feature matrix shape after scaling: {X_scaled.shape}")


# ---------- Train / test split ----------
test_size = 0.2 if FAST_MODE else 0.15
log(f"Train/test split (test_size={test_size})...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=test_size,
    random_state=42,
    stratify=y,
)
log(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ---------- Base model ----------
log("Building base RandomForest model (recall-focused)...")
base_rf = RandomForestClassifier(
    n_estimators=400 if FAST_MODE else 900,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    bootstrap=True,
    criterion="gini",
    class_weight="balanced",  # bias toward minority class to lift recall
    random_state=42,
    n_jobs=-1,
)


# ---------- Hyperparameter search ----------
log(f"Preparing RandomizedSearchCV (primary metric: {PRIMARY_METRIC})...")
cv_splits = 3 if FAST_MODE else 5
cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

param_distributions = {
    "n_estimators": [300, 400, 600, 900],
    "max_depth": [None, 10, 20, 40],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 3],
    "max_features": ["sqrt", "log2"],
}

n_iter = 10 if FAST_MODE else 25
log(f"RandomizedSearchCV: n_iter={n_iter}, cv={cv_splits}")

search = RandomizedSearchCV(
    estimator=base_rf,
    param_distributions=param_distributions,
    n_iter=n_iter,
    scoring=PRIMARY_METRIC,
    n_jobs=-1,
    cv=cv,
    random_state=42,
    verbose=1 if not FAST_MODE else 0,
)

t_search = time.time()
search.fit(X_train, y_train)
search_time = time.time() - t_search
log(f"RandomizedSearchCV done in {search_time:.2f}s")
log(f"Best params: {search.best_params_}")

best_model = search.best_estimator_


# ---------- Train best model on full train ----------
log("Training best RandomForest on full train set...")
t_fit = time.time()
best_model.fit(X_train, y_train)
fit_time = time.time() - t_fit


# ---------- Evaluate ----------
log("Evaluating on test set...")
y_pred = best_model.predict(X_test)

if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_test)[:, 1]
elif hasattr(best_model, "decision_function"):
    df_scores = best_model.decision_function(X_test)
    y_proba = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min() + 1e-12)
else:
    y_proba = None

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc_score = roc_auc_score(y_test, y_proba)

cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)


# ---------- Aggregate metrics ----------
results_df = pd.DataFrame(
    [
        {
            "Model": "RandomForest_Best",
            "Fit_Time_s": round(fit_time + search_time, 2),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUC": auc_score,
        }
    ]
)


# ---------- Save metrics ----------
excel_base = "AI_ByteA_ModelMetrics.xlsx"
excel_file = excel_base
log("Saving metrics to Excel...")
try:
    results_df.to_excel(excel_file, index=False, engine="openpyxl", sheet_name="Metrics")
except PermissionError:
    ts = time.strftime("%Y%m%d_%H%M%S")
    excel_file = f"AI_ByteA_ModelMetrics_{ts}.xlsx"
    log(f"Permission denied for '{excel_base}'. Saving instead to '{excel_file}'...")
    results_df.to_excel(excel_file, index=False, engine="openpyxl", sheet_name="Metrics")


# ---------- Save processed train set ----------
feature_names = [f"Feature_{i+1}" for i in range(X_train.shape[1])]
processed_df = pd.DataFrame(X_train, columns=feature_names)
processed_df[target_col] = y_train.values

proc_base = "AI_ByteA_ProcessedDataset.xlsx"
proc_file = proc_base
log("Saving processed train set to Excel...")
try:
    processed_df.to_excel(proc_file, index=False, engine="openpyxl")
except PermissionError:
    ts = time.strftime("%Y%m%d_%H%M%S")
    proc_file = f"AI_ByteA_ProcessedDataset_{ts}.xlsx"
    log(f"Permission denied for '{proc_base}'. Saving instead to '{proc_file}'...")
    processed_df.to_excel(proc_file, index=False, engine="openpyxl")


# ---------- Console summary ----------
log("Done.")
print(f"✓ RandomForest best model training completed in {time.time() - t0:.1f}s")
print(f"✓ Model Metrics → {excel_file}")
print(f"✓ Processed Train Dataset → {proc_file}")
print("\n" + "=" * 70)
print("BEST MODEL PERFORMANCE (RandomForest)")
print("=" * 70)
print(results_df.to_string(index=False))
print("=" * 70)
print("Confusion Matrix:")
print(cm)
print("=" * 70)

"""
Feature Engineering & Model Training Pipeline (Debug + Fast Mode)
Trains multiple ML models and generates performance metrics with progress logs
"""

import time, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as _e:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except Exception as e:
    SMOTE = None
    SMOTE_AVAILABLE = False
    print("WARNING: Could not import SMOTE from imbalanced-learn. "
          "Continuing without oversampling. Error:", e)

# --------------- helpers ---------------
def log(msg):
    print(time.strftime("[%H:%M:%S]"), msg)
    sys.stdout.flush()

t0 = time.time()
log("Start")

# ---------- SAFETY SWITCH ----------
FAST_MODE = True  # keep True for quick runs; set to False to scale up later

# ---------- plotting style ----------
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ---------- Load data ----------
log("Loading dataset...")
df = pd.read_excel('AI_ByteA_CleanedDataset.xlsx', engine='openpyxl')
log(f"Data shape: {df.shape}")

# ---------- Encode categoricals ----------
log("Encoding categoricals...")
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'PatientID':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# ---------- Target & features ----------
target_col = 'Diagnosis' if 'Diagnosis' in df.columns else df.columns[-1]
drop_cols = ['PatientID', target_col] if 'PatientID' in df.columns else [target_col]

X = df.drop(columns=drop_cols, errors='ignore')
y = df[target_col].astype(int)

# ---------- Impute ----------
log("Imputing missing values with KNNImputer...")
imputer = KNNImputer(n_neighbors=3 if FAST_MODE else 5)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ---------- Remove very low variance ----------
log("Variance thresholding...")
var_threshold = VarianceThreshold(threshold=0.0 if FAST_MODE else 0.0001)
X = pd.DataFrame(var_threshold.fit_transform(X), columns=X.columns[var_threshold.get_support()])

# ---------- Scale (no polynomial expansion; keep original feature space) ----------
log("Scaling features with StandardScaler (no polynomial features)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- (Optional) Feature selection ----------
# For now, keep all scaled features – tree-based models handle redundancy well,
# and aggressive selection was hurting test accuracy.
X_selected = X_scaled
log(f"Selected shape (no FS): {X_selected.shape}")

# ---------- Split ----------
test_size = 0.2 if FAST_MODE else 0.1
log(f"Train/test split (test_size={test_size})...")
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=test_size, random_state=42, stratify=y
)
log(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---------- Balance (use SMOTE if available) ----------
if SMOTE_AVAILABLE:
    log("Balancing with SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=3 if FAST_MODE else 5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    log(f"Balanced train: {X_train_bal.shape}, class ratio: {np.bincount(y_train_bal) if hasattr(np,'bincount') else 'n/a'}")
else:
    log("SMOTE not available – proceeding without oversampling (using original train set).")
    X_train_bal, y_train_bal = X_train, y_train

# ---------- Models (base configs) ----------
log("Building base models...")

models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=400 if FAST_MODE else 900,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        criterion='gini',
        class_weight='balanced',  # balanced for recall without hardcoding thresholds
        random_state=42,
        n_jobs=-1,
        verbose=0
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=300 if FAST_MODE else 700,
        learning_rate=0.05 if FAST_MODE else 0.02,
        max_depth=2,
        subsample=0.9,
        random_state=42
    ),
    'LogisticRegression': LogisticRegression(
        max_iter=3000 if FAST_MODE else 6000,
        C=1.0 if FAST_MODE else 3.0,
        penalty='l2',
        solver='lbfgs',
        class_weight='balanced',
        n_jobs=None,
        verbose=0
    ),
    'SVM': SVC(
        probability=True,
        kernel='rbf',
        C=1.0 if FAST_MODE else 3.0,
        gamma='scale',
        class_weight='balanced',
        cache_size=1024,
        tol=1e-3 if FAST_MODE else 1e-4,
        random_state=42
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        algorithm='auto',
        p=2
    ),
    'NaiveBayes': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    ),
}

if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBClassifier(
        n_estimators=300 if FAST_MODE else 600,
        max_depth=3 if FAST_MODE else 5,
        learning_rate=0.05 if FAST_MODE else 0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42,
        tree_method='hist'
    )
    log("XGBoost added to model list.")
else:
    log("XGBoost not available; skipping XGBoost model.")

# ---------- Lightweight hyperparameter tuning (RandomForest & GradientBoosting) ----------
log("Preparing hyperparameter search (accuracy-focused, limited for speed)...")
cv_splits = 3 if FAST_MODE else 5
cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

param_distributions = {
    'RandomForest': {
        'n_estimators': [300, 400, 600, 900],
        'max_depth': [None, 10, 20, 40],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2'],
    },
    'GradientBoosting': {
        'n_estimators': [200, 300, 500, 700],
        'learning_rate': [0.01, 0.03, 0.05],
        'max_depth': [2, 3, 4],
        'subsample': [0.8, 0.9, 1.0],
    },
}

if XGBOOST_AVAILABLE:
    param_distributions['XGBoost'] = {
        'n_estimators': [200, 300, 500],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.03, 0.05],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
    }

for m_name, grid in param_distributions.items():
    if m_name not in models:
        continue
    base_model = models[m_name]
    n_iter = 5 if FAST_MODE else 15
    log(f"Tuning {m_name} with RandomizedSearchCV (n_iter={n_iter}, cv={cv_splits})...")
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=grid,
        n_iter=n_iter,
        scoring='f1',  # tune for a balance of precision & recall
        n_jobs=-1,
        cv=cv,
        random_state=42,
        verbose=1 if not FAST_MODE else 0
    )
    search.fit(X_train_bal, y_train_bal)
    models[m_name] = search.best_estimator_
    log(f"{m_name} best params: {search.best_params_}")

# ---------- Train & evaluate ----------
log("Training tuned models & evaluating...")
results = []
confusion_matrices = {}
roc_data = {}

for name, model in models.items():
    t1 = time.time()
    log(f"Fitting {name}...")
    model.fit(X_train_bal, y_train_bal)
    fit_time = time.time() - t1

    # Base predictions at model's default threshold
    y_pred = model.predict(X_test)

    # Probabilities if available; else use decision_function or skip AUC/ROC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df_scores = model.decision_function(X_test)
        # scale to 0-1 for auc/roc
        y_proba = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min() + 1e-12)
    else:
        y_proba = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_score}

    results.append({
        'Model': name,
        'Fit_Time_s': round(fit_time, 2),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc_score
    })
    log(f"{name} done in {fit_time:.2f}s | AUC={auc_score:.3f}")

results_df = pd.DataFrame(results)
# Rank models by AUC (higher is better), similar to reference table
results_df['Rank'] = results_df['AUC'].rank(ascending=False, method='min').astype(int)

# Mark top-k models as selected (e.g., best 3 by AUC)
TOP_K_SELECTED = 3
results_df['Selected'] = (results_df['Rank'] <= TOP_K_SELECTED).astype(int)

# Reorder columns to mirror the reference: Rank, Model, Accuracy, F1, Precision, Recall, AUC, Selected
results_df = results_df[['Rank', 'Model', 'Accuracy', 'F1', 'Precision', 'Recall', 'AUC', 'Selected', 'Fit_Time_s']]

# ---------- Save metrics ----------
base_excel_file = 'AI_ByteA_ModelMetrics.xlsx'
excel_file = base_excel_file
log("Saving metrics to Excel...")
try:
    results_df.to_excel(excel_file, index=False, engine='openpyxl', sheet_name='Metrics')
except PermissionError:
    # if file is open/locked, write to a new file with timestamp
    ts = time.strftime("%Y%m%d_%H%M%S")
    excel_file = f"AI_ByteA_ModelMetrics_{ts}.xlsx"
    log(f"Permission denied for '{base_excel_file}'. Saving instead to '{excel_file}'...")
    results_df.to_excel(excel_file, index=False, engine='openpyxl', sheet_name='Metrics')

# ---------- Save processed train set ----------
feature_names = [f'Feature_{i+1}' for i in range(X_train_bal.shape[1])]
processed_df = pd.DataFrame(X_train_bal, columns=feature_names)
processed_df[target_col] = y_train_bal
proc_base = 'AI_ByteA_ProcessedDataset.xlsx'
proc_file = proc_base
try:
    processed_df.to_excel(proc_file, index=False, engine='openpyxl')
except PermissionError:
    ts = time.strftime("%Y%m%d_%H%M%S")
    proc_file = f"AI_ByteA_ProcessedDataset_{ts}.xlsx"
    log(f"Permission denied for '{proc_base}'. Saving processed dataset to '{proc_file}' instead...")
    processed_df.to_excel(proc_file, index=False, engine='openpyxl')

log("Done.")
print(f"✓ Model training completed in {time.time()-t0:.1f}s")
print(f"✓ Model Metrics with Embedded Graphs → {excel_file}")
print(f"✓ Processed Dataset → {proc_file}")
print("\n" + "="*70)
print("MODEL PERFORMANCE METRICS")
print("="*70)
print(results_df.to_string(index=False))
print("="*70)
