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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer, PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE

from openpyxl import load_workbook
from openpyxl.drawing.image import Image

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

# ---------- Scale/NORMALIZE ----------
log("Normalizing + scaling...")
normalizer = Normalizer()
X_norm = normalizer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_norm)

# ---------- Polynomial features (lightweight) ----------
if FAST_MODE:
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
else:
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)  # avoid degree=3 initially
log("Generating polynomial/interaction features...")
X_poly = poly.fit_transform(X_scaled)
log(f"Poly shape: {X_poly.shape}")

# ---------- Feature selection ----------
log("Selecting informative features (SelectKBest mutual_info)...")
k_cap = 100 if FAST_MODE else min(300, X_poly.shape[1])
selector = SelectKBest(mutual_info_classif, k=min(k_cap, X_poly.shape[1]-1))
X_selected = selector.fit_transform(X_poly, y)
log(f"Selected shape: {X_selected.shape}")

# ---------- Split ----------
test_size = 0.2 if FAST_MODE else 0.1
log(f"Train/test split (test_size={test_size})...")
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=test_size, random_state=42, stratify=y
)
log(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---------- Balance (use SMOTE only) ----------
log("Balancing with SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=3 if FAST_MODE else 5)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
log(f"Balanced train: {X_train_bal.shape}, class ratio: {np.bincount(y_train_bal) if hasattr(np,'bincount') else 'n/a'}")

# ---------- Models (scaled for speed) ----------
log("Building models...")

models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=300 if FAST_MODE else 800,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        criterion='gini',
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
        verbose=0
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200 if FAST_MODE else 500,
        learning_rate=0.05 if FAST_MODE else 0.03,
        max_depth=3,
        subsample=0.9,
        random_state=42
    ),
    'LogisticRegression': LogisticRegression(
        max_iter=2000 if FAST_MODE else 5000,
        C=1.0 if FAST_MODE else 3.0,
        penalty='l2',
        solver='lbfgs',
        class_weight='balanced',
        n_jobs=None,  # lbfgs ignores n_jobs
        verbose=0
    ),
    'SVM': SVC(
        probability=False,   # <<< main speed fix
        kernel='rbf',
        C=3.0 if FAST_MODE else 5.0,
        gamma='scale',
        class_weight='balanced',
        cache_size=1024,
        tol=1e-3 if FAST_MODE else 1e-4,
        random_state=42
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        algorithm='auto',
        p=2
    )
}

# ---------- Train & evaluate ----------
log("Training & evaluating...")
results = []
confusion_matrices = {}
roc_data = {}

for name, model in models.items():
    t1 = time.time()
    log(f"Fitting {name}...")
    model.fit(X_train_bal, y_train_bal)
    fit_time = time.time() - t1

    y_pred = model.predict(X_test)
    # Probabilities if available; else use decision_function or fallback to labels for AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df_scores = model.decision_function(X_test)
        # scale to 0-1 for auc/roc
        y_proba = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min() + 1e-12)
    else:
        y_proba = y_pred  # not ideal, but keeps pipeline working

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
results_df['Rank'] = results_df['AUC'].rank(ascending=False, method='min').astype(int)
results_df['Best_Model'] = (results_df['Rank'] == 1).astype(int)
results_df = results_df[['Model','Fit_Time_s','Accuracy','Precision','Recall','F1','AUC','Rank','Best_Model']]

# ---------- Save metrics ----------
excel_file = 'AI_ByteA_ModelMetrics.xlsx'
log("Saving metrics to Excel...")
results_df.to_excel(excel_file, index=False, engine='openpyxl', sheet_name='Metrics')

# ---------- Save processed train set ----------
feature_names = [f'Feature_{i+1}' for i in range(X_train_bal.shape[1])]
processed_df = pd.DataFrame(X_train_bal, columns=feature_names)
processed_df[target_col] = y_train_bal
processed_df.to_excel('AI_ByteA_ProcessedDataset.xlsx', index=False, engine='openpyxl')

# ---------- Confusion matrices figure ----------
log("Plotting confusion matrices...")
import math
rows = 2
cols = math.ceil((len(confusion_matrices)+1)/2)
fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
axes = np.array(axes).reshape(rows, cols)
fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')

i = 0
for name, cm in confusion_matrices.items():
    r, c = divmod(i, cols)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[r, c], cbar_kws={'label': 'Count'}, square=True)
    axes[r, c].set_title(name)
    axes[r, c].set_xlabel('Predicted')
    axes[r, c].set_ylabel('True')
    i += 1

# turn off extra axes
while i < rows*cols:
    r, c = divmod(i, cols)
    axes[r, c].axis('off')
    i += 1

plt.tight_layout()
plt.savefig('AI_ByteA_ConfusionMatrices.png', dpi=200, bbox_inches='tight')
plt.close()

# ---------- ROC curves ----------
log("Plotting ROC curves...")
plt.figure(figsize=(10, 7))
for name, data in roc_data.items():
    plt.plot(data['fpr'], data['tpr'], lw=2, label=f"{name} (AUC={data['auc']:.3f})")
plt.plot([0,1],[0,1],'k--', lw=1, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('AI_ByteA_ROC_Curves.png', dpi=200, bbox_inches='tight')
plt.close()

# ---------- Feature importance (GB) ----------
log("Plotting Gradient Boosting feature importances...")
gb_model = None
for k, v in models.items():
    if k == 'GradientBoosting':
        gb_model = v
        break

if gb_model is not None and hasattr(gb_model, 'feature_importances_'):
    fi = gb_model.feature_importances_
    top_n = min(20, len(fi))
    idx = np.argsort(fi)[-top_n:]
    plt.figure(figsize=(10, 7))
    plt.barh(range(top_n), fi[idx], alpha=0.9)
    plt.yticks(range(top_n), [f'Feature_{i+1}' for i in idx])
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Top Feature Importances - Gradient Boosting')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('AI_ByteA_GradientBoosting_FeatureImportance.png', dpi=200, bbox_inches='tight')
    plt.close()

# ---------- Embed images ----------
log("Embedding images into Excel...")
wb = load_workbook(excel_file)
ws = wb.active

img_cm = Image('AI_ByteA_ConfusionMatrices.png'); img_cm.width = 900; img_cm.height = 550
ws.add_image(img_cm, 'K2')

img_roc = Image('AI_ByteA_ROC_Curves.png'); img_roc.width = 600; img_roc.height = 400
ws.add_image(img_roc, 'K35')

try:
    img_fi = Image('AI_ByteA_GradientBoosting_FeatureImportance.png'); img_fi.width = 600; img_fi.height = 400
    ws.add_image(img_fi, 'K60')
except Exception as e:
    log(f"Feature importance image not added: {e}")

wb.save(excel_file)

log("Done.")
print(f"✓ Model training completed in {time.time()-t0:.1f}s")
print(f"✓ Model Metrics with Embedded Graphs → {excel_file}")
print(f"✓ Processed Dataset → AI_ByteA_ProcessedDataset.xlsx")
print("✓ Images:")
print("  - AI_ByteA_ConfusionMatrices.png")
print("  - AI_ByteA_ROC_Curves.png")
print("  - AI_ByteA_GradientBoosting_FeatureImportance.png")
print("\n" + "="*70)
print("MODEL PERFORMANCE METRICS")
print("="*70)
print(results_df.to_string(index=False))
print("="*70)
