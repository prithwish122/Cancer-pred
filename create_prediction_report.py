"""
Script to create a prediction report Excel file with:
- Best Model Used
- Performance Metrics
- Individual Predictions with Patient IDs
"""

import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

# Import the training pipeline components
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures

# Try to import SMOTE, but make it optional
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except (ImportError, TypeError):
    SMOTE_AVAILABLE = False
    print("Warning: SMOTE not available, will use class_weight balancing instead")

def create_prediction_report():
    """
    Creates an Excel file with model performance metrics and predictions
    """
    print("Loading dataset...")
    df = pd.read_excel('AI_ByteA_CleanedDataset.xlsx', engine='openpyxl')
    print(f"Data shape: {df.shape}")
    
    # Store PatientID if available
    patient_ids = None
    if 'PatientID' in df.columns:
        patient_ids = df['PatientID'].copy()
    
    # Encode categoricals
    print("Encoding categoricals...")
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'PatientID':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Target & features
    target_col = 'Diagnosis' if 'Diagnosis' in df.columns else df.columns[-1]
    drop_cols = ['PatientID', target_col] if 'PatientID' in df.columns else [target_col]
    
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col].astype(int)
    
    # Impute
    print("Imputing missing values...")
    imputer = KNNImputer(n_neighbors=3)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Variance thresholding
    print("Variance thresholding...")
    var_threshold = VarianceThreshold(threshold=0.0)
    X = pd.DataFrame(var_threshold.fit_transform(X), columns=X.columns[var_threshold.get_support()])
    
    # Scale/Normalize
    print("Normalizing + scaling...")
    normalizer = Normalizer()
    X_norm = normalizer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_norm)
    
    # Polynomial features
    print("Generating polynomial features...")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)
    print(f"Poly shape: {X_poly.shape}")
    
    # Feature selection
    print("Selecting informative features...")
    k_cap = 100
    selector = SelectKBest(mutual_info_classif, k=min(k_cap, X_poly.shape[1]-1))
    X_selected = selector.fit_transform(X_poly, y)
    print(f"Selected shape: {X_selected.shape}")
    
    # Split for model evaluation (to find best model)
    print("Train/test split for model evaluation...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Prepare full dataset for predictions (all data)
    X_full = X_selected
    y_full = y.values
    
    # Get all PatientIDs for full dataset
    if patient_ids is not None:
        patient_ids_full = patient_ids.values if hasattr(patient_ids, 'values') else list(patient_ids)
    else:
        # Generate PatientIDs if not available
        patient_ids_full = [f"PID{i+1}" for i in range(len(y_full))]
    
    print(f"Full dataset for predictions: {X_full.shape}")
    
    # Balance with SMOTE if available, otherwise use original data (models have class_weight)
    if SMOTE_AVAILABLE:
        try:
            print("Balancing with SMOTE...")
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
            print(f"Balanced train: {X_train_bal.shape}")
        except Exception as e:
            print(f"SMOTE failed ({e}), using original data with class_weight balancing")
            X_train_bal, y_train_bal = X_train, y_train
    else:
        print("Using class_weight balancing (SMOTE not available)")
        X_train_bal, y_train_bal = X_train, y_train
    
    # Models
    print("Building and training models...")
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
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
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=42
            # Note: GradientBoosting doesn't support class_weight, relies on SMOTE or natural balance
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=2000,
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            class_weight='balanced',
            n_jobs=None,
            verbose=0
        ),
        'SVM': SVC(
            probability=True,
            kernel='rbf',
            C=3.0,
            gamma='scale',
            class_weight='balanced',
            cache_size=1024,
            tol=1e-3,
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            algorithm='auto',
            p=2
        )
    }
    
    # Train & evaluate
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_bal, y_train_bal)
        trained_models[name] = model
        
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            df_scores = model.decision_function(X_test)
            y_proba = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min() + 1e-12)
        else:
            y_proba = y_pred.astype(float)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_score = roc_auc_score(y_test, y_proba)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc_score
        })
        print(f"{name} - AUC: {auc_score:.4f}")
    
    results_df = pd.DataFrame(results)
    
    # Find best model (highest AUC)
    best_model_name = results_df.loc[results_df['AUC'].idxmax(), 'Model']
    best_model = trained_models[best_model_name]
    best_metrics = results_df[results_df['Model'] == best_model_name].iloc[0]
    
    print(f"\nBest Model: {best_model_name}")
    
    # Retrain best model on FULL DATASET (with balancing) for final predictions
    print("Retraining best model on full dataset...")
    # Create a new instance of the best model to retrain
    best_model_final = models[best_model_name]
    
    # Balance full dataset if SMOTE available
    if SMOTE_AVAILABLE:
        try:
            print("Balancing full dataset with SMOTE...")
            smote_full = SMOTE(random_state=42, k_neighbors=3)
            X_full_bal, y_full_bal = smote_full.fit_resample(X_full, y_full)
            print(f"Balanced full dataset: {X_full_bal.shape}")
            best_model_final.fit(X_full_bal, y_full_bal)
        except Exception as e:
            print(f"SMOTE on full dataset failed ({e}), using original full dataset")
            best_model_final.fit(X_full, y_full)
    else:
        print("Training on full dataset with class_weight balancing...")
        best_model_final.fit(X_full, y_full)
    
    # Get predictions from best model on FULL DATASET
    print("Generating predictions for full dataset...")
    y_pred_full = best_model_final.predict(X_full)
    if hasattr(best_model_final, "predict_proba"):
        y_proba_full = best_model_final.predict_proba(X_full)[:, 1]
    else:
        y_proba_full = y_pred_full.astype(float)
    
    # Create predictions dataframe for full dataset
    predictions_df = pd.DataFrame({
        'PatientID': patient_ids_full,
        'Actual': y_full,
        'Predicted': y_pred_full,
        'Probability': y_proba_full,
        'Correct': (y_full == y_pred_full).astype(int)
    })
    
    # Create Excel file
    print("Creating Excel report...")
    wb = Workbook()
    ws = wb.active
    ws.title = "Prediction_Report"
    
    # Define styles
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    title_font = Font(bold=True, size=12)
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    center_align = Alignment(horizontal='center', vertical='center')
    
    # Section 1: Best Model Used
    ws['A1'] = "Best Model Used"
    ws['A1'].font = title_font
    ws['B1'] = best_model_name
    ws['B1'].font = Font(bold=True, size=11)
    
    # Section 2: Performance Metrics
    ws['A3'] = "Performance Metrics"
    ws.merge_cells('A3:G3')
    ws['A3'].font = title_font
    ws['A3'].alignment = center_align
    
    # Performance Metrics Headers
    headers = ['Model', '', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=4, column=col_idx)
        cell.value = header
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        cell.border = border
    
    # Performance Metrics Data (only best model)
    row = 5
    ws.cell(row=row, column=1).value = best_model_name
    ws.cell(row=row, column=3).value = round(best_metrics['Accuracy'], 3)
    ws.cell(row=row, column=4).value = round(best_metrics['Precision'], 6)
    ws.cell(row=row, column=5).value = round(best_metrics['Recall'], 3)
    ws.cell(row=row, column=6).value = round(best_metrics['F1'], 6)
    ws.cell(row=row, column=7).value = round(best_metrics['AUC'], 5)
    
    # Apply formatting to data row
    for col in range(1, 8):
        cell = ws.cell(row=row, column=col)
        cell.border = border
        if col == 1:
            cell.alignment = Alignment(horizontal='left')
        else:
            cell.alignment = center_align
    
    # Section 3: Predictions
    ws['A7'] = "Predictions"
    ws['A7'].font = title_font
    
    # Predictions Headers
    pred_headers = ['PatientID', '', 'Actual', 'Predicted', 'Probability', 'Correct']
    for col_idx, header in enumerate(pred_headers, start=1):
        cell = ws.cell(row=8, column=col_idx)
        cell.value = header
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        cell.border = border
    
    # Predictions Data
    for idx, (_, row_data) in enumerate(predictions_df.iterrows(), start=9):
        ws.cell(row=idx, column=1).value = str(row_data['PatientID'])
        ws.cell(row=idx, column=3).value = int(row_data['Actual'])
        ws.cell(row=idx, column=4).value = int(row_data['Predicted'])
        ws.cell(row=idx, column=5).value = round(row_data['Probability'], 6)
        ws.cell(row=idx, column=6).value = int(row_data['Correct'])
        
        # Apply formatting
        for col in range(1, 7):
            cell = ws.cell(row=idx, column=col)
            cell.border = border
            if col == 1:
                cell.alignment = Alignment(horizontal='left')
            else:
                cell.alignment = center_align
    
    # Adjust column widths
    ws.column_dimensions['A'].width = 15
    ws.column_dimensions['B'].width = 2
    ws.column_dimensions['C'].width = 10
    ws.column_dimensions['D'].width = 12
    ws.column_dimensions['E'].width = 12
    ws.column_dimensions['F'].width = 10
    ws.column_dimensions['G'].width = 10
    
    # Save file
    output_file = 'Prediction_Report.xlsx'
    wb.save(output_file)
    print(f"\n[SUCCESS] Prediction report saved to: {output_file}")
    print(f"[SUCCESS] Best Model: {best_model_name}")
    print(f"[SUCCESS] Total Predictions: {len(predictions_df)}")
    print(f"[SUCCESS] Correct Predictions: {predictions_df['Correct'].sum()}")
    print(f"[SUCCESS] Accuracy: {predictions_df['Correct'].mean():.4f}")
    
    return output_file

if __name__ == "__main__":
    create_prediction_report()

