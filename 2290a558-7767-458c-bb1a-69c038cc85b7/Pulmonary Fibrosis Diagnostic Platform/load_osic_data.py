import pandas as pd
import numpy as np
from pathlib import Path

# Download OSIC Pulmonary Fibrosis dataset using Kaggle API
# Since dataset is not present, we'll use the kaggle service to download it

print("OSIC Pulmonary Fibrosis dataset not found in data directory.")
print("The dataset needs to be downloaded from Kaggle first.")
print("\nDataset: osic-pulmonary-fibrosis-progression")
print("Source: https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression")
print("\nTo download, you can use the Kaggle API or the services/kaggle_service.py implemented in this canvas.")
print("\nFor demonstration purposes, I'll create a mock dataset with realistic structure:")

# Create realistic mock data based on OSIC competition structure
np.random.seed(42)

n_patients = 176  # Approximate number of patients in original dataset
n_weeks = [5, 10, 15, 20]  # Different patients have different numbers of visits

patient_records = []
patient_id_base = 'ID00'

for i in range(n_patients):
    patient_id = f'{patient_id_base}{i:03d}5637202311204720264'[:30]
    
    # Patient demographics
    age = np.random.randint(50, 85)
    sex = np.random.choice(['Male', 'Female'])
    smoking_status = np.random.choice(['Ex-smoker', 'Never smoked', 'Currently smokes'], p=[0.5, 0.3, 0.2])
    
    # Initial FVC measurement
    initial_fvc = np.random.uniform(1500, 4000)
    
    # Number of visits for this patient
    num_visits = np.random.choice([3, 5, 7, 10])
    
    # Generate visits over time
    weeks = sorted(np.random.choice(range(-12, 134), size=num_visits, replace=False))
    
    for week in weeks:
        # FVC declines over time with some noise
        fvc_decline_rate = np.random.uniform(5, 15)  # ml per week
        fvc = initial_fvc - (fvc_decline_rate * max(0, week)) + np.random.normal(0, 50)
        fvc = max(800, fvc)  # Minimum FVC
        
        # Percent is FVC as percentage of predicted
        percent = np.random.uniform(40, 90)
        
        patient_records.append({
            'Patient': patient_id,
            'Weeks': week,
            'FVC': round(fvc, 2),
            'Percent': round(percent, 2),
            'Age': age,
            'Sex': sex,
            'SmokingStatus': smoking_status
        })

train_df = pd.DataFrame(patient_records)

print(f"\n{'='*60}")
print("MOCK DATASET CREATED FOR DEMONSTRATION")
print('='*60)
print(f"Total records: {len(train_df):,}")
print(f"Total unique patients: {train_df['Patient'].nunique():,}")
print(f"\nDataset shape: {train_df.shape}")
print(f"\nColumn names and types:")
print(train_df.dtypes)
print(f"\nFirst few rows:")
print(train_df.head(10))
print(f"\nSample statistics:")
print(train_df.describe())
