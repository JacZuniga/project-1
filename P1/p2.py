import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = 'IPEDS_Default.csv'
df = pd.read_csv(file_path)

# Problem 1
# Drop NaN values for relevant columns
df_filtered = df[['DEGGRANT (HD2023)', 'DVADM01 (DRVADM2023)']].dropna()

# Create overlapping transparent histograms
plt.figure(figsize=(10, 6))
sns.histplot(df_filtered[df_filtered['DEGGRANT (HD2023)'] == 1]['DVADM01 (DRVADM2023)'], 
             color='blue', label='Degree-Granting', kde=True, alpha=0.5, bins=30)
sns.histplot(df_filtered[df_filtered['DEGGRANT (HD2023)'] == 2]['DVADM01 (DRVADM2023)'], 
             color='red', label='Non-Degree-Granting', kde=True, alpha=0.5, bins=30)

plt.title('Distribution of Admission Percentages by Degree-Granting Status', fontsize=18)
plt.xlabel('Percent Admitted - Total', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.legend()
plt.show()

# Problem 2:
# Drop NaN values for relevant columns
df_filtered2 = df[['DVADM01 (DRVADM2023)', 'GRRTTOT (DRVGR2023)']].dropna()

# Create a linear regression plot
plt.figure(figsize=(10, 6))
sns.regplot(x=df_filtered2['DVADM01 (DRVADM2023)'], y=df_filtered2['GRRTTOT (DRVGR2023)'], scatter_kws={'alpha':0.5})

plt.title('Relationship Between Admission Percentage and Graduation Rate', fontsize=18)
plt.xlabel('Percent Admitted - Total', fontsize=16)
plt.ylabel('Graduation Rate - Total Cohort', fontsize=16)
plt.show()