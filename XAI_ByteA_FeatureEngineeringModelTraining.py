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

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
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
# Primary objective: maximize recall (sensitivity) while still tracking accuracy/AUC
PRIMARY_METRIC = "recall"


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


# ---------- Hyperparameter search (GridSearchCV) ----------
log(f"Preparing GridSearchCV (primary metric: {PRIMARY_METRIC})...")
cv_splits = 3 if FAST_MODE else 5
cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

param_grid = {
    "n_estimators": [300, 500] if FAST_MODE else [300, 500, 800],
    "max_depth": [None, 15, 30],
    "min_samples_split": [2, 4],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
}

log(f"GridSearchCV over {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])} combinations, cv={cv_splits}")

search = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid,
    scoring={
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    },
    # Refit the model that maximizes cross-validated recall
    refit=PRIMARY_METRIC,
    n_jobs=-1,
    cv=cv,
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
log(f"Raw data shape: {df.shape}")

# ---------- Quick structural analysis (EDA) ----------
log(f"Number of duplicate rows before drop: {df.duplicated().sum()}")
log("Missing value ratio per column (top 10):")
log(df.isna().mean().sort_values(ascending=False).head(10).to_string())

# Basic description of numeric columns to inspect ranges / outliers
log("Numeric feature summary:")
log(df.select_dtypes(include=[np.number]).describe(percentiles=[0.01, 0.5, 0.99]).to_string())

# ---------- Basic cleaning: drop duplicates ----------
df = df.drop_duplicates().reset_index(drop=True)
log(f"Data shape after dropping duplicates: {df.shape}")

# ---------- Coerce numeric columns & cap extreme outliers ----------
num_cols_raw = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols_raw:
    # Coerce any bad formats to NaN to avoid silent string issues
    df[col] = pd.to_numeric(df[col], errors='coerce')

numeric_df = df[num_cols_raw]
q1 = numeric_df.quantile(0.01)
q99 = numeric_df.quantile(0.99)
# Winsorize numeric features so extreme outliers don't distort scaling
df[num_cols_raw] = numeric_df.clip(lower=q1, upper=q99, axis=1)

# ---------- Simple feature engineering (safe, recall-friendly) ----------
# Example: derive AgeBucket if Age exists to let models capture non-linear age-risk patterns
if 'Age' in df.columns:
    log("Creating AgeBucket feature from Age...")
    df['AgeBucket'] = pd.cut(
        df['Age'],
        bins=[0, 30, 45, 60, 75, 120],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    ).astype('int64')

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

# ---------- Train / test split (before fitting imputers/scalers to avoid leakage) ----------
test_size = 0.2 if FAST_MODE else 0.1
log(f"Train/test split (test_size={test_size})...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)
log(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---------- Impute (fit only on train, then transform test) ----------
log("Imputing missing values on train with KNNImputer (no leakage)...")
imputer = KNNImputer(n_neighbors=3 if FAST_MODE else 5)
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# ---------- Remove very low variance (fit on train only) ----------
log("Variance thresholding (fit on train only)...")
var_threshold = VarianceThreshold(threshold=0.0 if FAST_MODE else 0.0001)
X_train_var = var_threshold.fit_transform(X_train_imp)
X_test_var = var_threshold.transform(X_test_imp)

# ---------- Scale (fit scaler on train only to avoid test leakage) ----------
log("Scaling features with StandardScaler (fit on train only)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_var)
X_test_scaled = scaler.transform(X_test_var)

# ---------- PCA for dimensionality reduction (helps prevent overfitting) ----------
log("Applying PCA for dimensionality reduction (preserve ~95% variance)...")
# n_components=0.95 chooses the number of components that explain ~95% of variance
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
log(f"PCA-transformed shape: train={X_train_pca.shape}, test={X_test_pca.shape}")

# Downstream models will now use the PCA feature space
X_train, X_test = X_train_pca, X_test_pca

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

# ---------- Lightweight hyperparameter tuning (RandomForest, GradientBoosting, XGBoost) with GridSearchCV ----------
log("Preparing hyperparameter search with GridSearchCV (F1-focused, limited for speed)...")
cv_splits = 3 if FAST_MODE else 5
cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

param_grids = {
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
    param_grids['XGBoost'] = {
        'n_estimators': [200, 300, 500],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.03, 0.05],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8],
    }

for m_name, grid in param_grids.items():
    if m_name not in models:
        continue
    base_model = models[m_name]
    log(f"Tuning {m_name} with GridSearchCV (cv={cv_splits})...")
    search = GridSearchCV(
        estimator=base_model,
        param_grid=grid,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        # Choose the configuration with the highest cross-validated recall
        refit="recall",
        n_jobs=-1,
        cv=cv,
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

# Sort models by Recall (primary) and AUC (secondary) so ranking reflects real test performance
results_df = results_df.sort_values(['Recall', 'AUC'], ascending=[False, False]).reset_index(drop=True)
results_df['Rank'] = results_df.index + 1

# Reorder columns to: Rank, Model, Accuracy, F1, Precision, Recall, AUC, Fit_Time_s
results_df = results_df[['Rank', 'Model', 'Accuracy', 'F1', 'Precision', 'Recall', 'AUC', 'Fit_Time_s']]

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
