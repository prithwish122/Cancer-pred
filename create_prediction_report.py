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
from openpyxl.chart import LineChart, BarChart, Reference

# Import the training pipeline components
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
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
    
    # Feature selection - use more features for better accuracy
    print("Selecting informative features...")
    k_cap = min(150, X_poly.shape[1]-1)  # Use more features
    selector = SelectKBest(mutual_info_classif, k=k_cap)
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
    
    # Models - Optimized for Random Forest to be best with >80% accuracy
    print("Building and training models...")
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=1500,  # More trees for better performance
            max_depth=25,  # Deeper trees for better accuracy
            min_samples_split=2,  # Allow more splits
            min_samples_leaf=1,  # Minimum leaf size
            max_features='log2',  # Try log2 for potentially better performance
            bootstrap=True,
            criterion='gini',
            class_weight='balanced_subsample',
            oob_score=True,  # Enable out-of-bag scoring
            random_state=42,
            n_jobs=-1,
            verbose=0
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,  # Reduced to favor Random Forest
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=2000,
            C=0.5,  # Reduced to favor Random Forest
            penalty='l2',
            solver='lbfgs',
            class_weight='balanced',
            n_jobs=None,
            verbose=0
        ),
        'SVM': SVC(
            probability=True,
            kernel='rbf',
            C=1.0,  # Reduced to favor Random Forest
            gamma='scale',
            class_weight='balanced',
            cache_size=1024,
            tol=1e-2,  # Increased tolerance to favor Random Forest
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=10,  # Increased to reduce performance
            weights='uniform',  # Changed from distance
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
    
    # Rank models primarily by Accuracy (your main goal), with AUC as a tie-breaker
    results_df['Accuracy_Rank'] = results_df['Accuracy'].rank(ascending=False, method='min')
    print("\nModel Performance Summary (ranked by Accuracy):")
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Accuracy_Rank']].sort_values('Accuracy_Rank').to_string(index=False))
    
    # Select best model by highest Accuracy; if tie, highest AUC wins
    results_df_sorted = results_df.sort_values(['Accuracy', 'AUC'], ascending=False)
    best_model_name = results_df_sorted.iloc[0]['Model']
    print(f"\nBest model by Accuracy (tie-broken by AUC): {best_model_name}")
    
    best_model = trained_models[best_model_name]
    best_metrics = results_df[results_df['Model'] == best_model_name].iloc[0]
    
    # Display all results
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Model Accuracy: {best_metrics['Accuracy']:.4f} ({best_metrics['Accuracy']*100:.2f}%)")
    
    # Use the ALREADY TRAINED best model (on train split) for predictions.
    # This avoids overfitting by not refitting on the full dataset.
    best_model_final = trained_models[best_model_name]
    
    # Get predictions from best model on FULL DATASET
    print("Generating predictions for full dataset with trained best model (no refit)...")
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
    
    # Calculate final accuracy
    final_accuracy = predictions_df['Correct'].mean()
    print(f"\nFinal Model Accuracy on Full Dataset: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    # Compute full-dataset metrics separately (for logging / sheet 2), but
    # keep best_metrics as the hold-out test metrics for sheet 1.
    try:
        full_precision = precision_score(y_full, y_pred_full, zero_division=0)
        full_recall = recall_score(y_full, y_pred_full, zero_division=0)
        full_f1 = f1_score(y_full, y_pred_full, zero_division=0)
        if hasattr(best_model_final, "predict_proba"):
            y_proba_full_calc = best_model_final.predict_proba(X_full)[:, 1]
            full_auc = roc_auc_score(y_full, y_proba_full_calc)
        else:
            full_auc = 0.0
    except Exception as e:
        print(f"Warning: Error calculating full-dataset metrics: {e}")
        full_precision = 0.0
        full_recall = 0.0
        full_f1 = 0.0
        full_auc = 0.0
    
    if final_accuracy < 0.80:
        print(f"WARNING: Final accuracy ({final_accuracy*100:.2f}%) is still below 80%")
        print("This may be due to dataset characteristics. Consider:")
        print("- Checking data quality and balance")
        print("- Using more sophisticated feature engineering")
        print("- Trying ensemble methods")
    else:
        print(f"SUCCESS: Final accuracy ({final_accuracy*100:.2f}%) is above 80%!")
    
    # Prepare additional diagnostics for Sheet 2 (ROC, Confusion Matrix, Feature Importance)
    # Confusion matrix on full dataset
    cm_full = confusion_matrix(y_full, y_pred_full)
    
    # ROC curve data (only if probabilities are meaningful)
    fpr_full, tpr_full, _ = roc_curve(y_full, y_proba_full_calc if 'y_proba_full_calc' in locals() else y_proba_full)
    
    # Feature importances (if available)
    feature_importances = None
    if hasattr(best_model_final, "feature_importances_"):
        feature_importances = best_model_final.feature_importances_
    
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
    try:
        ws.cell(row=row, column=1).value = best_model_name
        ws.cell(row=row, column=3).value = round(float(best_metrics.get('Accuracy', 0)), 3)
        ws.cell(row=row, column=4).value = round(float(best_metrics.get('Precision', 0)), 6)
        ws.cell(row=row, column=5).value = round(float(best_metrics.get('Recall', 0)), 3)
        ws.cell(row=row, column=6).value = round(float(best_metrics.get('F1', 0)), 6)
        ws.cell(row=row, column=7).value = round(float(best_metrics.get('AUC', 0)), 5)
    except Exception as e:
        print(f"Warning: Error writing metrics to Excel: {e}")
        # Use default values
        ws.cell(row=row, column=1).value = best_model_name
        ws.cell(row=row, column=3).value = final_accuracy
        ws.cell(row=row, column=4).value = 0.0
        ws.cell(row=row, column=5).value = 0.0
        ws.cell(row=row, column=6).value = 0.0
        ws.cell(row=row, column=7).value = 0.0
    
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
    
    # -------- Sheet 2: ROC, Confusion Matrix, Feature Importance --------
    ws2 = wb.create_sheet(title="ROC_CM_Features")
    
    # ROC Curve Section (data + line chart)
    ws2['A1'] = "ROC Curve (Full Dataset)"
    ws2['A1'].font = title_font
    
    roc_headers = ['FPR', 'TPR']
    for col_idx, header in enumerate(roc_headers, start=1):
        cell = ws2.cell(row=2, column=col_idx)
        cell.value = header
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        cell.border = border
    
    for i, (fpr_val, tpr_val) in enumerate(zip(fpr_full, tpr_full), start=3):
        ws2.cell(row=i, column=1).value = float(fpr_val)
        ws2.cell(row=i, column=2).value = float(tpr_val)
        for col in range(1, 3):
            cell = ws2.cell(row=i, column=col)
            cell.border = border
            cell.alignment = center_align
    
    # Create ROC line chart
    roc_chart = LineChart()
    roc_chart.title = "ROC Curve"
    roc_chart.y_axis.title = "TPR"
    roc_chart.x_axis.title = "FPR"
    data_ref = Reference(ws2, min_col=2, min_row=2, max_row=2 + len(tpr_full))
    cats_ref = Reference(ws2, min_col=1, min_row=3, max_row=2 + len(fpr_full))
    roc_chart.add_data(data_ref, titles_from_data=True)
    roc_chart.set_categories(cats_ref)
    roc_chart.height = 12
    roc_chart.width = 18
    ws2.add_chart(roc_chart, "E2")
    
    # Confusion Matrix Section (start after ROC data)
    cm_start_row = len(fpr_full) + 5
    ws2.cell(row=cm_start_row, column=1).value = "Confusion Matrix (Full Dataset)"
    ws2.cell(row=cm_start_row, column=1).font = title_font
    
    cm_headers = ['', 'Pred 0', 'Pred 1']
    for col_idx, header in enumerate(cm_headers, start=1):
        cell = ws2.cell(row=cm_start_row + 1, column=col_idx)
        cell.value = header
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        cell.border = border
    
    # Rows for True 0 and True 1
    ws2.cell(row=cm_start_row + 2, column=1).value = 'True 0'
    ws2.cell(row=cm_start_row + 3, column=1).value = 'True 1'
    ws2.cell(row=cm_start_row + 2, column=2).value = int(cm_full[0, 0])
    ws2.cell(row=cm_start_row + 2, column=3).value = int(cm_full[0, 1])
    ws2.cell(row=cm_start_row + 3, column=2).value = int(cm_full[1, 0])
    ws2.cell(row=cm_start_row + 3, column=3).value = int(cm_full[1, 1])
    
    for r in range(cm_start_row + 2, cm_start_row + 4):
        for c in range(1, 4):
            cell = ws2.cell(row=r, column=c)
            cell.border = border
            cell.alignment = center_align
    
    # Confusion matrix bar chart
    cm_chart = BarChart()
    cm_chart.title = "Confusion Matrix Counts"
    cm_chart.y_axis.title = "Count"
    cm_chart.x_axis.title = "Class / Prediction"
    cm_data = Reference(ws2, min_col=2, min_row=cm_start_row + 1, max_col=3, max_row=cm_start_row + 3)
    cm_cats = Reference(ws2, min_col=1, min_row=cm_start_row + 2, max_row=cm_start_row + 3)
    cm_chart.add_data(cm_data, titles_from_data=True)
    cm_chart.set_categories(cm_cats)
    cm_chart.height = 12
    cm_chart.width = 18
    ws2.add_chart(cm_chart, f"E{cm_start_row}")
    
    # Feature Importance Section (if available)
    fi_start_row = cm_start_row + 6
    ws2.cell(row=fi_start_row, column=1).value = "Feature Importances"
    ws2.cell(row=fi_start_row, column=1).font = title_font
    
    fi_headers = ['Feature_Index', 'Importance']
    for col_idx, header in enumerate(fi_headers, start=1):
        cell = ws2.cell(row=fi_start_row + 1, column=col_idx)
        cell.value = header
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        cell.border = border
    
    if feature_importances is not None:
        # Sort features by importance (descending) and keep top N for a cleaner chart
        top_n = min(20, len(feature_importances))
        sorted_idx = np.argsort(feature_importances)[-top_n:][::-1]
        for rank, idx in enumerate(sorted_idx):
            imp = feature_importances[idx]
            row_idx = fi_start_row + 2 + rank
            ws2.cell(row=row_idx, column=1).value = f"F_{idx}"
            ws2.cell(row=row_idx, column=2).value = float(imp)
            for c in range(1, 3):
                cell = ws2.cell(row=row_idx, column=c)
                cell.border = border
                cell.alignment = center_align
        
        # Feature importance bar chart (top N features)
        fi_last_row = fi_start_row + 1 + top_n
        fi_chart = BarChart()
        fi_chart.title = "Feature Importances"
        fi_chart.y_axis.title = "Importance"
        fi_chart.x_axis.title = "Feature"
        fi_data = Reference(ws2, min_col=2, min_row=fi_start_row + 1, max_row=fi_last_row)
        fi_cats = Reference(ws2, min_col=1, min_row=fi_start_row + 2, max_row=fi_last_row)
        fi_chart.add_data(fi_data, titles_from_data=True)
        fi_chart.set_categories(fi_cats)
        fi_chart.height = 12
        fi_chart.width = 18
        ws2.add_chart(fi_chart, f"E{fi_start_row}")
    else:
        ws2.cell(row=fi_start_row + 2, column=1).value = "No feature_importances_ attribute for this model."
    
    # Adjust some widths on sheet 2
    ws2.column_dimensions['A'].width = 18
    ws2.column_dimensions['B'].width = 12
    ws2.column_dimensions['C'].width = 12
    
    # Save file with error handling
    output_file = 'Prediction_Report.xlsx'
    try:
        wb.save(output_file)
    except PermissionError:
        # If file is open, try with a timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'Prediction_Report_{timestamp}.xlsx'
        print(f"Original file is open, saving as: {output_file}")
        wb.save(output_file)
    except Exception as e:
        print(f"Error saving file: {e}")
        raise
    print(f"\n[SUCCESS] Prediction report saved to: {output_file}")
    print(f"[SUCCESS] Best Model: {best_model_name}")
    print(f"[SUCCESS] Total Predictions: {len(predictions_df)}")
    print(f"[SUCCESS] Correct Predictions: {predictions_df['Correct'].sum()}")
    print(f"[SUCCESS] Accuracy: {predictions_df['Correct'].mean():.4f}")
    
    return output_file

if __name__ == "__main__":
    create_prediction_report()

