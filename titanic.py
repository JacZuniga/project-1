import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind

# Load the Titanic dataset
file_path = "/Users/jacktorres/Desktop/B DATA 200/titanic.csv"
df = pd.read_csv(file_path)

# Keep only relevant columns
df_filtered = df[['Age', 'Sex', 'Survived']].copy()

# Remove rows with missing Age values instead of imputing
df_filtered = df_filtered.dropna(subset=['Age'])

# Define age groups
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+']
df_filtered['Age Group'] = pd.cut(df_filtered['Age'], bins=bins, labels=labels, right=False)

# Compute survival rates by gender and age group
survival_rates = df_filtered.groupby(['Age Group', 'Sex'])['Survived'].mean().unstack()

# Chi-Square test for gender-based survival differences
contingency_table = pd.crosstab(df_filtered['Sex'], df_filtered['Survived'])
chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

# Determine at what age male survival significantly drops
male_data = df_filtered[df_filtered['Sex'] == 'male']
female_data = df_filtered[df_filtered['Sex'] == 'female']

# Perform t-test between age groups to determine statistical significance in survival differences
age_10_20 = male_data[male_data['Age Group'] == '10-20']['Survived']
age_20_30 = male_data[male_data['Age Group'] == '20-30']['Survived']

t_stat, t_p_value = ttest_ind(age_10_20, age_20_30, equal_var=False, nan_policy='omit')

# Plot survival rates
plt.figure(figsize=(10, 5))
ax = survival_rates.plot(kind='bar', figsize=(10, 5))
plt.title('Survival Rates by Age Group and Gender')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)
plt.legend(title='Sex')

# Add sample size labels
for i, age_group in enumerate(labels):
    total_male = df_filtered[(df_filtered['Age Group'] == age_group) & (df_filtered['Sex'] == 'male')].shape[0]
    total_female = df_filtered[(df_filtered['Age Group'] == age_group) & (df_filtered['Sex'] == 'female')].shape[0]
    ax.text(i - 0.15, survival_rates.loc[age_group, 'male'] + 0.02, f'n={total_male}', color='blue', fontsize=10)
    ax.text(i + 0.05, survival_rates.loc[age_group, 'female'] + 0.02, f'n={total_female}', color='red', fontsize=10)

plt.show()

# Output results
print(f"Chi-Square Statistic: {chi2_stat:.2f}")
print(f"p-value: {p_value:.5f} (Extremely significant gender effect)")

print(f"T-test Statistic (10-20 vs. 20-30 years): {t_stat:.2f}")
print(f"p-value: {t_p_value:.5f} (Significance of male survival drop between 10-20 and 20-30 years)")