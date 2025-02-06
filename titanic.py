import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load the Titanic dataset
file_path = "/Users/jacktorres/Desktop/B DATA 200/titanic.csv"
df = pd.read_csv(file_path)

# Keep only relevant columns
df_filtered = df[['Age', 'Sex', 'Survived']].copy()

# Handle missing values by filling with median age
df_filtered['Age'].fillna(df_filtered['Age'].median(), inplace=True)

# Bin ages into groups for easier analysis
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
age_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']
df_filtered['AgeGroup'] = pd.cut(df_filtered['Age'], bins=age_bins, labels=age_labels, right=False)

# Calculate survival rates by age group and gender
survival_rates = df_filtered.groupby(['AgeGroup', 'Sex'])['Survived'].mean().reset_index()

# Set style for visualization
sns.set_style("whitegrid")

# Create a bar plot for survival rates by age group and gender
plt.figure(figsize=(10, 6))
sns.barplot(x='AgeGroup', y='Survived', hue='Sex', data=survival_rates, palette=['blue', 'pink'])

# Labels and title
plt.xlabel("Age Group", fontsize=12)
plt.ylabel("Survival Rate", fontsize=12)
plt.title("Survival Rate by Age Group and Gender on the Titanic", fontsize=14)
plt.ylim(0, 1)
plt.legend(title="Sex")

# Show plot
plt.xticks(rotation=45)
plt.show()

# Perform Chi-square test on survival by gender
contingency_table = pd.crosstab(df_filtered['Sex'], df_filtered['Survived'])
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

# Output results
print(f"Chi-square Statistic: {chi2_stat}")
print(f"P-value: {p_value}")
