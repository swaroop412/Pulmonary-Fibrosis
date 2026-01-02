import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Analyze patient demographics

print("="*60)
print("PATIENT DEMOGRAPHICS ANALYSIS")
print("="*60)

# Get unique patient demographics (one row per patient)
patient_demographics = train_df.groupby('Patient').first()[['Age', 'Sex', 'SmokingStatus']].reset_index()

# Age distribution
print(f"\nAge Statistics:")
print(f"  Mean age: {patient_demographics['Age'].mean():.1f} years")
print(f"  Median age: {patient_demographics['Age'].median():.1f} years")
print(f"  Age range: {patient_demographics['Age'].min()}-{patient_demographics['Age'].max()} years")
print(f"  Std dev: {patient_demographics['Age'].std():.1f} years")

# Age distribution by bins
age_bins = pd.cut(patient_demographics['Age'], bins=[40, 55, 65, 75, 90], labels=['50-55', '56-65', '66-75', '76+'])
print(f"\nAge Distribution:")
print(age_bins.value_counts().sort_index())

# Sex distribution
print(f"\nSex Distribution:")
sex_counts = patient_demographics['Sex'].value_counts()
print(sex_counts)
print(f"  Male percentage: {sex_counts['Male']/len(patient_demographics)*100:.1f}%")
print(f"  Female percentage: {sex_counts.get('Female', 0)/len(patient_demographics)*100:.1f}%")

# Smoking status distribution
print(f"\nSmoking Status Distribution:")
smoking_counts = patient_demographics['SmokingStatus'].value_counts()
print(smoking_counts)
for status, count in smoking_counts.items():
    print(f"  {status}: {count/len(patient_demographics)*100:.1f}%")

# Age by sex
print(f"\nAge by Sex:")
print(patient_demographics.groupby('Sex')['Age'].describe())

# Age by smoking status
print(f"\nAge by Smoking Status:")
print(patient_demographics.groupby('SmokingStatus')['Age'].describe())

# Create demographic visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#1D1D20')

colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF']

# Age distribution histogram
ax = axes[0, 0]
ax.set_facecolor('#1D1D20')
ax.hist(patient_demographics['Age'], bins=15, color='#A1C9F4', edgecolor='#fbfbff', alpha=0.8)
ax.set_xlabel('Age (years)', fontsize=11, color='#fbfbff')
ax.set_ylabel('Number of Patients', fontsize=11, color='#fbfbff')
ax.set_title('Age Distribution of Patients', fontsize=13, color='#fbfbff', fontweight='bold', pad=15)
ax.tick_params(colors='#fbfbff')
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

# Sex distribution pie chart
ax = axes[0, 1]
ax.set_facecolor('#1D1D20')
wedges, texts, autotexts = ax.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%',
                                    colors=['#A1C9F4', '#FFB482'], startangle=90,
                                    textprops={'color': '#fbfbff', 'fontsize': 11})
ax.set_title('Sex Distribution', fontsize=13, color='#fbfbff', fontweight='bold', pad=15)

# Smoking status bar chart
ax = axes[1, 0]
ax.set_facecolor('#1D1D20')
bars = ax.bar(range(len(smoking_counts)), smoking_counts.values, color=colors[:len(smoking_counts)], 
               edgecolor='#fbfbff', alpha=0.8)
ax.set_xticks(range(len(smoking_counts)))
ax.set_xticklabels(smoking_counts.index, rotation=15, ha='right', fontsize=10, color='#fbfbff')
ax.set_ylabel('Number of Patients', fontsize=11, color='#fbfbff')
ax.set_title('Smoking Status Distribution', fontsize=13, color='#fbfbff', fontweight='bold', pad=15)
ax.tick_params(colors='#fbfbff')
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

# Age by sex boxplot
ax = axes[1, 1]
ax.set_facecolor('#1D1D20')
sex_groups = [patient_demographics[patient_demographics['Sex'] == s]['Age'].values for s in sex_counts.index]
bp = ax.boxplot(sex_groups, labels=sex_counts.index, patch_artist=True,
                boxprops=dict(facecolor='#A1C9F4', alpha=0.8),
                medianprops=dict(color='#ffd400', linewidth=2),
                whiskerprops=dict(color='#fbfbff'),
                capprops=dict(color='#fbfbff'))
ax.set_ylabel('Age (years)', fontsize=11, color='#fbfbff')
ax.set_title('Age Distribution by Sex', fontsize=13, color='#fbfbff', fontweight='bold', pad=15)
ax.tick_params(colors='#fbfbff')
for spine in ax.spines.values():
    spine.set_edgecolor('#909094')

plt.tight_layout()
demographics_fig = fig
plt.show()

print(f"\n✓ Demographics analysis complete with {len(patient_demographics)} unique patients")
