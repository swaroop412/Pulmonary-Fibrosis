import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt

# Laboratory Data Preprocessing Pipeline
# Processing FVC, Age, and Smoking Status from the OSIC dataset

print("=" * 70)
print("LABORATORY DATA PREPROCESSING PIPELINE")
print("=" * 70)

# Load the training data
print("\n📊 LOADING OSIC DATASET")
print("-" * 70)
print(f"Dataset shape: {train_df.shape}")
print(f"Columns: {list(train_df.columns)}")

# Data cleaning and quality assessment
print("\n\n🔍 DATA QUALITY ASSESSMENT")
print("-" * 70)

# Check for missing values
missing_data = train_df.isnull().sum()
print("Missing values per column:")
for col, count in missing_data.items():
    pct = (count / len(train_df)) * 100
    print(f"  {col}: {count} ({pct:.2f}%)")

# Check for outliers using IQR method
def detect_outliers_iqr(df, column):
    """Detect outliers using Interquartile Range method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

print("\n\n🎯 OUTLIER DETECTION")
print("-" * 70)

outlier_summary = {}
for col in ['FVC', 'Percent', 'Age']:
    outliers, lower, upper = detect_outliers_iqr(train_df, col)
    outlier_summary[col] = {
        'count': len(outliers),
        'percentage': (len(outliers) / len(train_df)) * 100,
        'lower_bound': lower,
        'upper_bound': upper
    }
    print(f"{col}:")
    print(f"  Outliers: {len(outliers)} ({(len(outliers)/len(train_df)*100):.2f}%)")
    print(f"  Valid range: [{lower:.2f}, {upper:.2f}]")

# Create cleaned dataset
lab_data_cleaned = train_df.copy()

# Handle any missing values (forward fill within patient groups)
lab_data_cleaned = lab_data_cleaned.sort_values(['Patient', 'Weeks'])
lab_data_cleaned['FVC'] = lab_data_cleaned.groupby('Patient')['FVC'].fillna(method='ffill')
lab_data_cleaned['Percent'] = lab_data_cleaned.groupby('Patient')['Percent'].fillna(method='ffill')

# Feature engineering for laboratory data
print("\n\n⚙️ FEATURE ENGINEERING")
print("-" * 70)

# 1. Baseline measurements (first measurement for each patient)
baseline_data = lab_data_cleaned.groupby('Patient').first().reset_index()
baseline_features = baseline_data[['Patient', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']].copy()
baseline_features.columns = ['Patient', 'Baseline_FVC', 'Baseline_Percent', 'Age', 'Sex', 'SmokingStatus']

# 2. FVC decline rate for each patient
fvc_slopes = []
for patient in lab_data_cleaned['Patient'].unique():
    patient_data = lab_data_cleaned[lab_data_cleaned['Patient'] == patient]
    if len(patient_data) > 1:
        # Calculate slope using linear regression
        weeks = patient_data['Weeks'].values
        fvc_values = patient_data['FVC'].values
        slope = np.polyfit(weeks, fvc_values, 1)[0]
    else:
        slope = 0
    fvc_slopes.append({'Patient': patient, 'FVC_Decline_Rate': slope})

fvc_decline_df = pd.DataFrame(fvc_slopes)

# 3. Merge engineered features
lab_features = baseline_features.merge(fvc_decline_df, on='Patient')

# 4. Encode categorical variables
# Sex encoding
lab_features['Sex_Encoded'] = lab_features['Sex'].map({'Male': 1, 'Female': 0})

# Smoking status encoding (ordinal: Never < Ex < Current)
smoking_map = {'Never smoked': 0, 'Ex-smoker': 1, 'Currently smokes': 2}
lab_features['SmokingStatus_Encoded'] = lab_features['SmokingStatus'].map(smoking_map)

# 5. Create one-hot encoding for smoking status
lab_features_onehot = pd.get_dummies(lab_features, columns=['SmokingStatus'], prefix='Smoking')

print("Engineered features:")
print(f"  • Baseline FVC and Percent")
print(f"  • FVC decline rate (slope)")
print(f"  • Sex encoding (binary)")
print(f"  • Smoking status encoding (ordinal and one-hot)")
print(f"\nTotal features created: {lab_features_onehot.shape[1]}")

# Normalization strategies
print("\n\n📏 NORMALIZATION STRATEGIES")
print("-" * 70)

# Prepare numerical columns for scaling
numerical_cols = ['Baseline_FVC', 'Baseline_Percent', 'Age', 'FVC_Decline_Rate']

# 1. Standard Scaler (z-score normalization)
standard_scaler = StandardScaler()
lab_standard_scaled = lab_features[numerical_cols].copy()
lab_standard_scaled[numerical_cols] = standard_scaler.fit_transform(lab_features[numerical_cols])

# 2. Min-Max Scaler (0-1 normalization)
minmax_scaler = MinMaxScaler()
lab_minmax_scaled = lab_features[numerical_cols].copy()
lab_minmax_scaled[numerical_cols] = minmax_scaler.fit_transform(lab_features[numerical_cols])

# 3. Robust Scaler (median and IQR - robust to outliers)
robust_scaler = RobustScaler()
lab_robust_scaled = lab_features[numerical_cols].copy()
lab_robust_scaled[numerical_cols] = robust_scaler.fit_transform(lab_features[numerical_cols])

print("Available normalization methods:")
print("  1. StandardScaler: Mean=0, Std=1 (best for normally distributed data)")
print("  2. MinMaxScaler: Range=[0,1] (best for bounded features)")
print("  3. RobustScaler: Median/IQR (best for data with outliers)")

# Compare scaling results
print("\n\n📊 SCALING COMPARISON")
print("-" * 70)
scaling_comparison = pd.DataFrame({
    'Feature': numerical_cols,
    'Original_Mean': lab_features[numerical_cols].mean().values,
    'Original_Std': lab_features[numerical_cols].std().values,
    'Standard_Mean': lab_standard_scaled.mean().values,
    'Standard_Std': lab_standard_scaled.std().values,
    'MinMax_Min': lab_minmax_scaled.min().values,
    'MinMax_Max': lab_minmax_scaled.max().values
})
print(scaling_comparison.to_string(index=False))

# Final preprocessed dataset
print("\n\n✅ FINAL PREPROCESSED LABORATORY DATA")
print("-" * 70)

# Create final dataset with all features and normalized values
lab_data_final = lab_features_onehot.copy()

# Add normalized features (using StandardScaler as default)
for col in numerical_cols:
    lab_data_final[f'{col}_normalized'] = standard_scaler.fit_transform(lab_features[[col]])

print(f"Final dataset shape: {lab_data_final.shape}")
print(f"Total features: {lab_data_final.shape[1]}")
print(f"Number of patients: {len(lab_data_final)}")

print("\n📋 Feature summary:")
print(lab_data_final.head())

print("\n📈 Descriptive statistics:")
print(lab_data_final[numerical_cols].describe())

# Configuration summary
preprocessing_config = {
    'missing_value_strategy': 'Forward fill within patient groups',
    'outlier_handling': 'IQR detection (kept for now, can be removed if needed)',
    'feature_engineering': [
        'Baseline measurements (first visit per patient)',
        'FVC decline rate (linear slope)',
        'Categorical encoding (Sex, Smoking Status)',
        'One-hot encoding for smoking status'
    ],
    'normalization_method': 'StandardScaler (z-score)',
    'alternative_scalers': ['MinMaxScaler', 'RobustScaler'],
    'final_features': list(lab_data_final.columns),
    'data_ready': True
}

print("\n\n🎯 PREPROCESSING PIPELINE CONFIGURATION")
print("-" * 70)
print(f"Missing value strategy: {preprocessing_config['missing_value_strategy']}")
print(f"Normalization method: {preprocessing_config['normalization_method']}")
print(f"Total features in final dataset: {len(preprocessing_config['final_features'])}")

print("\n✅ LABORATORY DATA PREPROCESSING COMPLETE - MODEL READY")
print("=" * 70)

# Store preprocessing artifacts
lab_preprocessing_complete = True
total_patients_processed = len(lab_data_final)
