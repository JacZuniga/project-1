import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency

# Set the working directory
dir_name = '/Users/jacktorres/Dropbox/Mac/Desktop/B DATA 200/p1'
os.chdir(dir_name)
working_dir = os.getcwd()
print(working_dir)

# Load the dataset
file_path = 'IPEDS_Default.csv'
ipeds_df = pd.read_csv(file_path)
print(ipeds_df.head())

"""
Part A: Use a t-test to compare “Total price for in-state students living on 
campus 2023-24 (DRVIC2023)” (short variable name CINSON (DRVIC2023)) 
for Historically Black Colleges or Universities (HBCUs) and non-HBCU institutions 
(short variable name HBCU (HD2023)). Report the p value and your assumptions.
"""

# Filter the data for HBCUs and non-HBCUs
hbcus = ipeds_df[ipeds_df['HBCU (HD2023)'] == 1]['CINSON (DRVIC2023)']
non_hbcus = ipeds_df[ipeds_df['HBCU (HD2023)'] == 2]['CINSON (DRVIC2023)']

# Perform the t-test
t_statistic, p_value = stats.ttest_ind(hbcus, non_hbcus, nan_policy='omit')

# Print the results
print("t-statistic:", t_statistic)
print("P-value:", p_value)

# Test the hypothesis
alpha = 0.05
if p_value < alpha:
    print("Reject null hypothesis: There is a significant difference between the means.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the means.")

"""
Part B: Provide the mean and standard deviation of the total price for both.
"""

# Calculate and print the mean and standard deviation for HBCUs
mean_hbcus = np.mean(hbcus)
std_dev_hbcus = np.std(hbcus)
print("Mean for HBCUs (CINSON (DRVIC2023)): ", mean_hbcus)
print("Standard Deviation for HBCUs (CINSON (DRVIC2023)): ", std_dev_hbcus)

# Calculate and print the mean and standard deviation for non-HBCUs
mean_non_hbcus = np.mean(non_hbcus)
std_dev_non_hbcus = np.std(non_hbcus)
print("Mean for non-HBCUs (CINSON (DRVIC2023)): ", mean_non_hbcus)
print("Standard Deviation for non-HBCUs (CINSON (DRVIC2023)): ", std_dev_non_hbcus)

"""
Part A: Do a linear regression between the “Total 12-month unduplicated headcount 
(DRVEF122023)” (short variable name UNDUP (DRVEF122023)) and “Number of students 
receiving a Doctor's degree (DRVC2023)” (short variable name SDOCDEG (DRVC2023)). 
Provide the slope, intercept, and r value. Do these variables correlate?
"""

# Rename columns for easier access (adjust column names if incorrect)
ipeds_df.rename(columns={'UNDUP (DRVEF122023)': 'headcount', 
                         'SDOCDEG (DRVC2023)': 'doctorates'}, inplace=True)

# Convert to numeric and handle missing values
ipeds_df['headcount'] = pd.to_numeric(ipeds_df['headcount'], errors='coerce')
ipeds_df['doctorates'] = pd.to_numeric(ipeds_df['doctorates'], errors='coerce')

# Drop NaN values
df_clean = ipeds_df.dropna(subset=['headcount', 'doctorates'])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['headcount'], df_clean['doctorates'])

# Print results
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-value (correlation): {r_value}")
print(f"P-value: {p_value}")
print(f"Standard error: {std_err}")

# Interpretation
if abs(r_value) > 0.7:
    print("Strong correlation between total headcount and number of doctorate recipients.")
elif abs(r_value) > 0.4:
    print("Moderate correlation between total headcount and number of doctorate recipients.")
else:
    print("Weak or no correlation between total headcount and number of doctorate recipients.")

ipeds_df.rename(columns={'TRIBAL (HD2023)': 'tribal', 'DEGGRANT (HD2023)': 'degree_granting'}, inplace=True)

# Convert to numeric
ipeds_df['tribal'] = pd.to_numeric(ipeds_df['tribal'], errors='coerce')
ipeds_df['degree_granting'] = pd.to_numeric(ipeds_df['degree_granting'], errors='coerce')

# Create contingency table
contingency_table = pd.crosstab(ipeds_df['tribal'], ipeds_df['degree_granting'])

# Perform Chi-Squared Test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

# Print results
print("Chi-Squared Statistic:", chi2_stat)
print("P-value:", p_value)
print("Degrees of Freedom:", dof)
print("Expected Frequencies Table:\n", expected)

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Tribal College status is associated with Degree-Granting Status.")
else:
    print("Fail to reject the null hypothesis: Tribal College status is independent of Degree-Granting Status.")



