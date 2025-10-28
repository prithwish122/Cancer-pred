"""
Data Cleaning Script for AI_ByteA_DirtyDataset
Handles duplicates, missing values, outliers, and data quality issues
"""

import pandas as pd
import numpy as np

# Load dirty dataset
df = pd.read_excel('AI_ByteA_DirtyDataset.xlsx', engine='openpyxl')

# Step 1: Drop duplicate rows
df = df.drop_duplicates()

# Step 2: Identify and separate column types before standardization
# Preserve column names for important fields
preserved_cols = df.columns.tolist()

# Step 3: Handle missing values
# Numeric columns: fill with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert invalid entries to NaN
    df[col].fillna(df[col].median(), inplace=True)

# Categorical columns: fill with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if not df[col].mode().empty:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Step 4: Remove outliers using IQR method (only for medical/numeric columns, not Age, PatientID)
outlier_cols = [col for col in numeric_cols if col not in ['PatientID', 'Age']]
for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Step 5: Clean Age column (remove invalid ages)
if 'Age' in df.columns:
    df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]

# Step 6: Reset index after filtering
df = df.reset_index(drop=True)

# Save cleaned dataset (preserve all original columns)
df.to_excel('AI_ByteA_CleanedDataset.xlsx', index=False, engine='openpyxl')
print(f"✓ Cleaned dataset saved: {len(df)} records → AI_ByteA_CleanedDataset.xlsx")
