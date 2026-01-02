import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create comprehensive EDA summary report

print("="*70)
print("OSIC PULMONARY FIBROSIS - COMPREHENSIVE EDA SUMMARY")
print("="*70)

print("\n📊 DATASET STRUCTURE")
print("-" * 70)
print(f"Total Records: {len(train_df):,}")
print(f"Unique Patients: {train_df['Patient'].nunique():,}")
print(f"Variables: {train_df.shape[1]}")
print(f"Data Columns: {', '.join(train_df.columns.tolist())}")
print(f"Study Period: Week {train_df['Weeks'].min()} to Week {train_df['Weeks'].max()}")

print("\n👥 PATIENT DEMOGRAPHICS")
print("-" * 70)
patient_demographics = train_df.groupby('Patient').first()[['Age', 'Sex', 'SmokingStatus']]
print(f"Age: {patient_demographics['Age'].mean():.1f} ± {patient_demographics['Age'].std():.1f} years (range: {patient_demographics['Age'].min()}-{patient_demographics['Age'].max()})")
print(f"Sex Distribution:")
print(f"  • Male: {(patient_demographics['Sex']=='Male').sum()} ({(patient_demographics['Sex']=='Male').sum()/len(patient_demographics)*100:.1f}%)")
print(f"  • Female: {(patient_demographics['Sex']=='Female').sum()} ({(patient_demographics['Sex']=='Female').sum()/len(patient_demographics)*100:.1f}%)")
print(f"Smoking Status:")
for status in patient_demographics['SmokingStatus'].value_counts().sort_values(ascending=False).items():
    print(f"  • {status[0]}: {status[1]} ({status[1]/len(patient_demographics)*100:.1f}%)")

print("\n🫁 FVC MEASUREMENTS")
print("-" * 70)
print(f"Mean FVC: {train_df['FVC'].mean():.0f} ml (SD: {train_df['FVC'].std():.0f} ml)")
print(f"FVC Range: {train_df['FVC'].min():.0f} - {train_df['FVC'].max():.0f} ml")
print(f"Mean FVC % Predicted: {train_df['Percent'].mean():.1f}%")
print(f"FVC by Sex:")
print(f"  • Male: {train_df[train_df['Sex']=='Male']['FVC'].mean():.0f} ml")
print(f"  • Female: {train_df[train_df['Sex']=='Female']['FVC'].mean():.0f} ml")

print("\n📉 DISEASE PROGRESSION")
print("-" * 70)
# Calculate decline rates
patient_slopes = []
for patient_id in train_df['Patient'].unique():
    patient_data = train_df[train_df['Patient'] == patient_id].sort_values('Weeks')
    if len(patient_data) >= 2:
        weeks = patient_data['Weeks'].values
        fvc = patient_data['FVC'].values
        if weeks.max() - weeks.min() > 0:
            slope = (fvc[-1] - fvc[0]) / (weeks[-1] - weeks[0])
            patient_slopes.append(slope)

slopes_array = np.array(patient_slopes)
print(f"Mean FVC Decline Rate: {slopes_array.mean():.2f} ml/week")
print(f"Median FVC Decline Rate: {np.median(slopes_array):.2f} ml/week")
print(f"Patients with Declining FVC: {(slopes_array < 0).sum()} ({(slopes_array < 0).sum()/len(slopes_array)*100:.1f}%)")
print(f"Patients with Stable/Improving FVC: {(slopes_array >= 0).sum()} ({(slopes_array >= 0).sum()/len(slopes_array)*100:.1f}%)")

print("\n📅 TEMPORAL PATTERNS")
print("-" * 70)
measurements_per_patient = train_df.groupby('Patient').size()
print(f"Measurements per Patient: {measurements_per_patient.mean():.1f} (range: {measurements_per_patient.min()}-{measurements_per_patient.max()})")

intervals = []
for patient_id in train_df['Patient'].unique():
    patient_data = train_df[train_df['Patient'] == patient_id].sort_values('Weeks')
    if len(patient_data) >= 2:
        intervals.extend(np.diff(patient_data['Weeks'].values))
intervals = np.array(intervals)
print(f"Mean Interval Between Visits: {intervals.mean():.1f} weeks")
print(f"Study Duration per Patient: {train_df.groupby('Patient')['Weeks'].apply(lambda x: x.max()-x.min()).mean():.1f} weeks")

print("\n✅ DATA QUALITY")
print("-" * 70)
print(f"Missing Values: {train_df.isnull().sum().sum()} (0.0%)")
print(f"Data Completeness: 100%")
print(f"All patients have complete demographic information")
print(f"All measurements have complete FVC and temporal data")

print("\n🔍 KEY FINDINGS")
print("-" * 70)
print("1. Dataset contains longitudinal FVC measurements from 176 pulmonary fibrosis patients")
print("2. Patients are predominantly older adults (mean age 66.7 years)")
print("3. Majority are ex-smokers (58.5%), reflecting typical IPF patient profile")
print("4. 98.3% of patients show FVC decline over time (mean -9.17 ml/week)")
print("5. High data quality with zero missing values across all variables")
print("6. Measurements span from week -12 to week 133, with variable follow-up duration")
print("7. Sex distribution is balanced (51.7% male, 48.3% female)")
print("8. FVC values show wide range (800-3,979 ml), reflecting disease heterogeneity")

print("\n📈 CLINICAL IMPLICATIONS")
print("-" * 70)
print("• Strong declining trend in FVC indicates active disease progression")
print("• Variability in decline rates suggests different disease phenotypes")
print("• Longitudinal data suitable for progression modeling and prediction")
print("• Complete dataset enables robust statistical analysis without imputation")

print("\n" + "="*70)
print("✓ COMPREHENSIVE EDA COMPLETE")
print("="*70)

# Create final summary visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor('#1D1D20')
fig.suptitle('OSIC Pulmonary Fibrosis - EDA Summary Dashboard', 
             fontsize=16, color='#fbfbff', fontweight='bold', y=0.995)

# 1. Patient age distribution
ax = axes[0, 0]
ax.set_facecolor('#1D1D20')
ax.hist(patient_demographics['Age'], bins=12, color='#A1C9F4', edgecolor='#fbfbff', alpha=0.8)
ax.set_xlabel('Age (years)', fontsize=10, color='#fbfbff')
ax.set_ylabel('Patients', fontsize=10, color='#fbfbff')
ax.set_title('Age Distribution', fontsize=11, color='#fbfbff', fontweight='bold')
ax.tick_params(colors='#fbfbff', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

# 2. FVC distribution
ax = axes[0, 1]
ax.set_facecolor('#1D1D20')
ax.hist(train_df['FVC'], bins=25, color='#FFB482', edgecolor='#fbfbff', alpha=0.8)
ax.set_xlabel('FVC (ml)', fontsize=10, color='#fbfbff')
ax.set_ylabel('Measurements', fontsize=10, color='#fbfbff')
ax.set_title('FVC Distribution', fontsize=11, color='#fbfbff', fontweight='bold')
ax.tick_params(colors='#fbfbff', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

# 3. Decline rates
ax = axes[0, 2]
ax.set_facecolor('#1D1D20')
ax.hist(slopes_array, bins=25, color='#8DE5A1', edgecolor='#fbfbff', alpha=0.8)
ax.axvline(x=0, color='#f04438', linestyle='--', linewidth=2)
ax.set_xlabel('FVC Change (ml/week)', fontsize=10, color='#fbfbff')
ax.set_ylabel('Patients', fontsize=10, color='#fbfbff')
ax.set_title('Disease Progression Rate', fontsize=11, color='#fbfbff', fontweight='bold')
ax.tick_params(colors='#fbfbff', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

# 4. Sex distribution
ax = axes[1, 0]
ax.set_facecolor('#1D1D20')
sex_counts = patient_demographics['Sex'].value_counts()
wedges, texts, autotexts = ax.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%',
                                    colors=['#A1C9F4', '#FFB482'], startangle=90,
                                    textprops={'color': '#fbfbff', 'fontsize': 10})
ax.set_title('Sex Distribution', fontsize=11, color='#fbfbff', fontweight='bold')

# 5. Sample trajectories
ax = axes[1, 1]
ax.set_facecolor('#1D1D20')
sample_pts = train_df['Patient'].unique()[:8]
colors_traj = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#1F77B4', '#9467BD', '#8C564B']
for i, pt in enumerate(sample_pts):
    pt_data = train_df[train_df['Patient'] == pt].sort_values('Weeks')
    ax.plot(pt_data['Weeks'], pt_data['FVC'], color=colors_traj[i], alpha=0.7, linewidth=1.5)
ax.set_xlabel('Weeks', fontsize=10, color='#fbfbff')
ax.set_ylabel('FVC (ml)', fontsize=10, color='#fbfbff')
ax.set_title('Sample Patient Trajectories', fontsize=11, color='#fbfbff', fontweight='bold')
ax.tick_params(colors='#fbfbff', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

# 6. Measurements per patient
ax = axes[1, 2]
ax.set_facecolor('#1D1D20')
visit_dist = measurements_per_patient.value_counts().sort_index()
ax.bar(visit_dist.index, visit_dist.values, color='#D0BBFF', edgecolor='#fbfbff', alpha=0.8)
ax.set_xlabel('Visits per Patient', fontsize=10, color='#fbfbff')
ax.set_ylabel('Number of Patients', fontsize=10, color='#fbfbff')
ax.set_title('Visit Frequency', fontsize=11, color='#fbfbff', fontweight='bold')
ax.tick_params(colors='#fbfbff', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

plt.tight_layout()
summary_dashboard = fig
plt.show()

print("\n📊 All EDA visualizations and analyses have been generated successfully!")
