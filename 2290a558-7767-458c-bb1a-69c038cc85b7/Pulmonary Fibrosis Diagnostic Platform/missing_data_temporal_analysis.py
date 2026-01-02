import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Missing value and temporal pattern analysis

print("="*60)
print("MISSING VALUE & TEMPORAL PATTERN ANALYSIS")
print("="*60)

# Missing value analysis
print("\nMissing Values by Column:")
missing_counts = train_df.isnull().sum()
missing_pct = (train_df.isnull().sum() / len(train_df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing_Count': missing_counts,
    'Missing_Percentage': missing_pct
})
print(missing_df)

if missing_df['Missing_Count'].sum() == 0:
    print("\n✓ No missing values detected in the dataset!")
else:
    print(f"\nTotal missing values: {missing_df['Missing_Count'].sum()}")

# Data completeness per patient
patient_completeness = train_df.groupby('Patient').apply(
    lambda x: (x.notna().sum().sum() / (len(x) * len(train_df.columns))) * 100
)
print(f"\nData Completeness per Patient:")
print(f"  Mean completeness: {patient_completeness.mean():.2f}%")
print(f"  Median completeness: {patient_completeness.median():.2f}%")
print(f"  Min completeness: {patient_completeness.min():.2f}%")
print(f"  Max completeness: {patient_completeness.max():.2f}%")

# Temporal patterns - time between measurements
print("\n" + "="*60)
print("TEMPORAL MEASUREMENT PATTERNS")
print("="*60)

intervals = []
for patient_id in train_df['Patient'].unique():
    patient_data = train_df[train_df['Patient'] == patient_id].sort_values('Weeks')
    if len(patient_data) >= 2:
        patient_intervals = np.diff(patient_data['Weeks'].values)
        intervals.extend(patient_intervals)

intervals = np.array(intervals)
print(f"\nTime Between Consecutive Measurements:")
print(f"  Mean interval: {intervals.mean():.1f} weeks")
print(f"  Median interval: {np.median(intervals):.1f} weeks")
print(f"  Min interval: {intervals.min()} weeks")
print(f"  Max interval: {intervals.max()} weeks")
print(f"  Std dev: {intervals.std():.1f} weeks")

# Study duration per patient
study_durations = []
baseline_weeks = []
for patient_id in train_df['Patient'].unique():
    patient_data = train_df[train_df['Patient'] == patient_id].sort_values('Weeks')
    duration = patient_data['Weeks'].max() - patient_data['Weeks'].min()
    study_durations.append(duration)
    baseline_weeks.append(patient_data['Weeks'].min())

print(f"\nStudy Duration per Patient:")
print(f"  Mean duration: {np.mean(study_durations):.1f} weeks")
print(f"  Median duration: {np.median(study_durations):.1f} weeks")
print(f"  Range: {np.min(study_durations)}-{np.max(study_durations)} weeks")

print(f"\nBaseline Visit Week:")
print(f"  Mean baseline: {np.mean(baseline_weeks):.1f}")
print(f"  Median baseline: {np.median(baseline_weeks):.1f}")
print(f"  Range: {np.min(baseline_weeks)}-{np.max(baseline_weeks)}")

# Patient cohort analysis
early_measurements = train_df[train_df['Weeks'] < 0]
baseline_measurements = train_df[(train_df['Weeks'] >= 0) & (train_df['Weeks'] < 20)]
mid_measurements = train_df[(train_df['Weeks'] >= 20) & (train_df['Weeks'] < 80)]
late_measurements = train_df[train_df['Weeks'] >= 80]

print(f"\nMeasurement Time Cohorts:")
print(f"  Pre-baseline (Week < 0): {len(early_measurements)} measurements ({len(early_measurements)/len(train_df)*100:.1f}%)")
print(f"  Baseline (Week 0-20): {len(baseline_measurements)} measurements ({len(baseline_measurements)/len(train_df)*100:.1f}%)")
print(f"  Mid-study (Week 20-80): {len(mid_measurements)} measurements ({len(mid_measurements)/len(train_df)*100:.1f}%)")
print(f"  Late-study (Week 80+): {len(late_measurements)} measurements ({len(late_measurements)/len(train_df)*100:.1f}%)")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#1D1D20')

# Measurement intervals distribution
ax = axes[0, 0]
ax.set_facecolor('#1D1D20')
ax.hist(intervals, bins=30, color='#A1C9F4', edgecolor='#fbfbff', alpha=0.8)
ax.set_xlabel('Weeks Between Measurements', fontsize=11, color='#fbfbff')
ax.set_ylabel('Frequency', fontsize=11, color='#fbfbff')
ax.set_title('Distribution of Measurement Intervals', fontsize=13, color='#fbfbff', fontweight='bold', pad=15)
ax.tick_params(colors='#fbfbff')
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

# Study duration per patient
ax = axes[0, 1]
ax.set_facecolor('#1D1D20')
ax.hist(study_durations, bins=25, color='#FFB482', edgecolor='#fbfbff', alpha=0.8)
ax.set_xlabel('Study Duration (weeks)', fontsize=11, color='#fbfbff')
ax.set_ylabel('Number of Patients', fontsize=11, color='#fbfbff')
ax.set_title('Patient Study Duration Distribution', fontsize=13, color='#fbfbff', fontweight='bold', pad=15)
ax.tick_params(colors='#fbfbff')
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

# Measurements per patient bar chart
ax = axes[1, 0]
ax.set_facecolor('#1D1D20')
visit_counts = train_df.groupby('Patient').size().value_counts().sort_index()
ax.bar(visit_counts.index, visit_counts.values, color='#8DE5A1', edgecolor='#fbfbff', alpha=0.8)
ax.set_xlabel('Number of Measurements', fontsize=11, color='#fbfbff')
ax.set_ylabel('Number of Patients', fontsize=11, color='#fbfbff')
ax.set_title('Distribution of Measurements per Patient', fontsize=13, color='#fbfbff', fontweight='bold', pad=15)
ax.tick_params(colors='#fbfbff')
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

# Temporal cohort comparison
ax = axes[1, 1]
ax.set_facecolor('#1D1D20')
cohort_labels = ['Pre-baseline\n(< 0)', 'Baseline\n(0-20)', 'Mid-study\n(20-80)', 'Late-study\n(80+)']
cohort_counts = [len(early_measurements), len(baseline_measurements), len(mid_measurements), len(late_measurements)]
colors_cohorts = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B']
bars = ax.bar(range(len(cohort_labels)), cohort_counts, color=colors_cohorts, edgecolor='#fbfbff', alpha=0.8)
ax.set_xticks(range(len(cohort_labels)))
ax.set_xticklabels(cohort_labels, fontsize=10, color='#fbfbff')
ax.set_ylabel('Number of Measurements', fontsize=11, color='#fbfbff')
ax.set_title('Measurements by Study Phase', fontsize=13, color='#fbfbff', fontweight='bold', pad=15)
ax.tick_params(colors='#fbfbff')
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

plt.tight_layout()
temporal_fig = fig
plt.show()

print(f"\n✓ Missing value and temporal pattern analysis complete")
