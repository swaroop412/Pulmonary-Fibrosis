import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Analyze FVC progression patterns

print("="*60)
print("FVC PROGRESSION ANALYSIS")
print("="*60)

# Overall FVC statistics
print(f"\nFVC (Forced Vital Capacity) Statistics:")
print(f"  Mean FVC: {train_df['FVC'].mean():.2f} ml")
print(f"  Median FVC: {train_df['FVC'].median():.2f} ml")
print(f"  FVC range: {train_df['FVC'].min():.2f} - {train_df['FVC'].max():.2f} ml")
print(f"  Std dev: {train_df['FVC'].std():.2f} ml")

# FVC Percent statistics
print(f"\nFVC Percent (% of predicted) Statistics:")
print(f"  Mean: {train_df['Percent'].mean():.2f}%")
print(f"  Median: {train_df['Percent'].median():.2f}%")
print(f"  Range: {train_df['Percent'].min():.2f}% - {train_df['Percent'].max():.2f}%")

# Temporal statistics
print(f"\nTemporal Measurement Statistics:")
print(f"  Week range: {train_df['Weeks'].min()} to {train_df['Weeks'].max()}")
print(f"  Mean week: {train_df['Weeks'].mean():.1f}")
print(f"  Total measurements: {len(train_df)}")

# Measurements per patient
measurements_per_patient = train_df.groupby('Patient').size()
print(f"\nMeasurements per Patient:")
print(f"  Mean: {measurements_per_patient.mean():.1f}")
print(f"  Median: {measurements_per_patient.median():.1f}")
print(f"  Range: {measurements_per_patient.min()}-{measurements_per_patient.max()}")
print(f"\nDistribution of visit counts:")
print(measurements_per_patient.value_counts().sort_index())

# Calculate decline rates for patients with multiple visits
patient_slopes = []
for patient_id in train_df['Patient'].unique():
    patient_data = train_df[train_df['Patient'] == patient_id].sort_values('Weeks')
    if len(patient_data) >= 2:
        # Calculate slope (FVC change per week)
        weeks = patient_data['Weeks'].values
        fvc = patient_data['FVC'].values
        if weeks.max() - weeks.min() > 0:
            slope = (fvc[-1] - fvc[0]) / (weeks[-1] - weeks[0])
            patient_slopes.append({
                'Patient': patient_id,
                'Slope': slope,
                'Initial_FVC': fvc[0],
                'Final_FVC': fvc[-1],
                'Duration': weeks[-1] - weeks[0]
            })

slopes_df = pd.DataFrame(patient_slopes)
print(f"\nFVC Decline Rates (ml/week):")
print(f"  Mean decline: {slopes_df['Slope'].mean():.2f} ml/week")
print(f"  Median decline: {slopes_df['Slope'].median():.2f} ml/week")
print(f"  Patients with decline: {(slopes_df['Slope'] < 0).sum()} ({(slopes_df['Slope'] < 0).sum()/len(slopes_df)*100:.1f}%)")
print(f"  Patients with improvement: {(slopes_df['Slope'] > 0).sum()} ({(slopes_df['Slope'] > 0).sum()/len(slopes_df)*100:.1f}%)")

# FVC by demographics
print(f"\nFVC by Sex:")
print(train_df.groupby('Sex')['FVC'].agg(['mean', 'median', 'std']))

print(f"\nFVC by Smoking Status:")
print(train_df.groupby('SmokingStatus')['FVC'].agg(['mean', 'median', 'std']))

# Create FVC progression visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#1D1D20')

# Sample patient trajectories
ax = axes[0, 0]
ax.set_facecolor('#1D1D20')
sample_patients = train_df['Patient'].unique()[:10]
colors_palette = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#1F77B4', '#9467BD', '#8C564B', '#C49C94', '#E377C2']
for i, patient_id in enumerate(sample_patients):
    patient_data = train_df[train_df['Patient'] == patient_id].sort_values('Weeks')
    ax.plot(patient_data['Weeks'], patient_data['FVC'], marker='o', 
            color=colors_palette[i], alpha=0.7, linewidth=2, markersize=5)
ax.set_xlabel('Weeks', fontsize=11, color='#fbfbff')
ax.set_ylabel('FVC (ml)', fontsize=11, color='#fbfbff')
ax.set_title('Sample Patient FVC Trajectories (n=10)', fontsize=13, color='#fbfbff', fontweight='bold', pad=15)
ax.tick_params(colors='#fbfbff')
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

# FVC distribution
ax = axes[0, 1]
ax.set_facecolor('#1D1D20')
ax.hist(train_df['FVC'], bins=30, color='#A1C9F4', edgecolor='#fbfbff', alpha=0.8)
ax.set_xlabel('FVC (ml)', fontsize=11, color='#fbfbff')
ax.set_ylabel('Frequency', fontsize=11, color='#fbfbff')
ax.set_title('FVC Distribution', fontsize=13, color='#fbfbff', fontweight='bold', pad=15)
ax.tick_params(colors='#fbfbff')
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

# Decline rate distribution
ax = axes[1, 0]
ax.set_facecolor('#1D1D20')
ax.hist(slopes_df['Slope'], bins=30, color='#FFB482', edgecolor='#fbfbff', alpha=0.8)
ax.axvline(x=0, color='#f04438', linestyle='--', linewidth=2, label='No change')
ax.set_xlabel('FVC Change Rate (ml/week)', fontsize=11, color='#fbfbff')
ax.set_ylabel('Number of Patients', fontsize=11, color='#fbfbff')
ax.set_title('FVC Decline Rate Distribution', fontsize=13, color='#fbfbff', fontweight='bold', pad=15)
ax.tick_params(colors='#fbfbff')
ax.legend(facecolor='#1D1D20', edgecolor='#909094', labelcolor='#fbfbff')
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

# Measurements over time
ax = axes[1, 1]
ax.set_facecolor('#1D1D20')
week_counts = train_df['Weeks'].value_counts().sort_index()
ax.bar(week_counts.index, week_counts.values, color='#8DE5A1', edgecolor='#fbfbff', alpha=0.8, width=3)
ax.set_xlabel('Week', fontsize=11, color='#fbfbff')
ax.set_ylabel('Number of Measurements', fontsize=11, color='#fbfbff')
ax.set_title('Temporal Distribution of Measurements', fontsize=13, color='#fbfbff', fontweight='bold', pad=15)
ax.tick_params(colors='#fbfbff')
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

plt.tight_layout()
fvc_fig = fig
plt.show()

print(f"\n✓ FVC progression analysis complete")
