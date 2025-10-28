"""
Synthetic Patient Data Generator with Intentional Anomalies
Uses SDV, Faker, NumPy, and Pandas to generate dirty healthcare data
"""

import pandas as pd
import numpy as np
from faker import Faker
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from pathlib import Path

# Initialize
fake = Faker()
np.random.seed(42)
Faker.seed(42)

# Load input data
input_file = Path('The_Cancer_data_1500_V2.csv')
df_original = pd.read_csv(input_file)

# Auto-detect metadata and train synthesizer
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_original)
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(df_original)

# Generate 1000 synthetic records
df_synthetic = synthesizer.sample(num_rows=1000)

# Add/overwrite demographic columns using vectorized operations
df_synthetic['PatientID'] = [f'P{str(i+1).zfill(4)}' for i in range(1000)]
df_synthetic['Name'] = [fake.name() for _ in range(1000)]
df_synthetic['Gender'] = np.random.choice(['Male', 'Female', 'Other'], size=1000, p=[0.48, 0.48, 0.04])
df_synthetic['Age'] = np.random.randint(18, 85, size=1000)
df_synthetic['Race'] = np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Native American', 'Other'], size=1000)
df_synthetic['Ethnicity'] = np.random.choice(['Hispanic or Latino', 'Not Hispanic or Latino'], size=1000, p=[0.18, 0.82])
df_synthetic['Occupation'] = [fake.job() for _ in range(1000)]
df_synthetic['DietType'] = np.random.choice(['Omnivore', 'Vegetarian', 'Vegan', 'Pescatarian', 'Keto', 'Mediterranean'], size=1000)
df_synthetic['City'] = [fake.city() for _ in range(1000)]
df_synthetic['State'] = [fake.state() for _ in range(1000)]
df_synthetic['Country'] = np.random.choice(['USA', 'Canada', 'UK', 'Australia'], size=1000, p=[0.70, 0.15, 0.10, 0.05])

# Remove CancerHistory column if it exists
if 'CancerHistory' in df_synthetic.columns:
    df_synthetic = df_synthetic.drop(columns=['CancerHistory'])

# Reorder columns: PatientID, Name, Gender, Age, Race, Ethnicity, Occupation, DietType, City, State, Country, then rest
priority_cols = ['PatientID', 'Name', 'Gender', 'Age', 'Race', 'Ethnicity', 'Occupation', 'DietType', 'City', 'State', 'Country']
other_cols = [col for col in df_synthetic.columns if col not in priority_cols]
df_synthetic = df_synthetic[priority_cols + other_cols]

# Introduce anomalies
# 1. Random nulls (5% across random columns)
null_mask = np.random.rand(*df_synthetic.shape) < 0.05
df_synthetic = df_synthetic.mask(null_mask)

# 2. Duplicates (50 random rows duplicated)
dup_indices = np.random.choice(df_synthetic.index, size=50, replace=False)
df_synthetic = pd.concat([df_synthetic, df_synthetic.loc[dup_indices]], ignore_index=True)

# 3. Age outliers (20 invalid ages)
outlier_indices = np.random.choice(df_synthetic.index, size=20, replace=False)
df_synthetic.loc[outlier_indices, 'Age'] = np.random.choice([-5, -1, 0, 150, 300, 999], size=20)

# 4. Wrong data formats (corrupt numeric columns with strings if they exist)
numeric_cols = df_synthetic.select_dtypes(include=[np.number]).columns.difference(['Age'])
if len(numeric_cols) > 0:
    corrupt_col = numeric_cols[0]
    corrupt_indices = np.random.choice(df_synthetic.index, size=15, replace=False)
    corrupt_values = np.random.choice(['invalid', 'N/A', 'error', '#REF!', 'null'], size=15)
    df_synthetic.loc[corrupt_indices, corrupt_col] = corrupt_values

# Save to Excel
df_synthetic.to_excel('AI_ByteA_DirtyDataset.xlsx', index=False, engine='openpyxl')
print(f"✓ Generated {len(df_synthetic)} records → AI_ByteA_DirtyDataset.xlsx")
