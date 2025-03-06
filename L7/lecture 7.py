import os
import scipy 
import seaborn as sns
import joypy
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Chaning working directorory 
directory_name = '/Users/jacktorres/Dropbox/Mac/Desktop/B DATA 200/L7'
os.chdir(directory_name)
working_dir = os.getcwd()
print(working_dir)

# Opening file 
file_name = "titanic copy.csv"
titanicDF = pd.read_csv(file_name)
print(titanicDF.head())

print("Default Shape")
print(titanicDF.shape)
print(titanicDF.columns)
print("Default Index")
print(titanicDF.index)
titanicDF = titanicDF.set_index(list(titanicDF)[0])
print("First Row Set as Index")
print(titanicDF.index)
print("New Shape")
print(titanicDF.shape)

print("New Dataframe")
print(titanicDF.head())

print(titanicDF.loc[1, 'Fare'])
print(titanicDF.iloc[0, 8])
print(titanicDF['Fare'][1])

print("Is the first passanger's fare NA?")
print(pd.isna(titanicDF.loc[1, 'Fare']))
print("Is the first passanger's Cabin NA?")
print(pd.isna(titanicDF.loc[1, 'Cabin']))

print("Titanic first 400 passangers")
titanic400 = titanicDF.iloc[0:400, :]
print(titanic400.tail())
print("Shape of first 400 passangers")
print(titanic400.shape)

print("Titanic, Age and Cabin are not NaN")
age_fare = titanicDF[pd.notna(titanicDF["Age"])]
age_fare = age_fare[pd.notna(titanicDF["Cabin"])]
print("Original List, first 10 rows")
print(titanicDF.head(10))
print("Age and fare are not NA, first 10 rows")
print(age_fare.head(10))

print("Mean Age")
print(titanicDF['Age'].mean(skipna=True))
print("Mean age with NAs")
print(titanicDF['Age'].mean(skipna=False))
print("Median Age")
print(titanicDF['Age'].median(skipna=True))
print("Mode Age")
print(titanicDF['Age'].mode(dropna=True))
discstat = titanicDF['Age'].describe(percentiles = [0.1, 0.9])
print(discstat)
print("90th percentile plus one year")
print(discstat["90%"] + 1)

print("Name, sex, and age of the last 10 passengers")
vitals_last_10 = titanicDF.iloc[-10:-1,:]
vitals_last_10 = vitals_last_10.loc[:,'Name':'Age']

print("Size and contents of vitals_last_10")
print(vitals_last_10.shape)
print(vitals_last_10.tail())

array_1 = [1, 2, 3]
array_2 = [5, 6, 7]

t_test_res = scipy.stats.ttest_ind(array_1, array_2, equal_var = False, nan_policy
= 'omit', alternative = 'less')

print("T test result")
print(t_test_res)

print("Chi Squared Example")
example_observed = [[10, 10, 20], [20, 20, 20]]
chi_squared_results = scipy.stats.chi2_contingency(example_observed)

print(chi_squared_results)

print("Linear Regression Example")
x_pigs = [1, 2, 3, 4, 5, 6, 7]
y_pigs = [2, 2, 8, 7, 7, 3, 9]

regress_pigs = scipy.stats.linregress(x_pigs, y_pigs, alternative = 'two-sided')
print(regress_pigs)

# Making a Histogram in Python
plt.hist(titanicDF["Age"], bins = 10)
plt.gca().set(title = "Ages in the Titanic Dataset", ylabel = "Count", xlabel =
"Age (years)")

plt.show()
survivorDF = titanicDF[titanicDF["Survived"] == 1]
diedDF = titanicDF[titanicDF["Survived"] == 0]

matplotlib.rcParams.update({'font.size': 18})

plt.hist(titanicDF['Age'], bins = 10)
plt.gca().set(title = "Ages in the Titanic Dataset", ylabel = "Count",
xlabel = "Age (years)")
plt.show()

# Overlapping Historgrams (Age, survivors and non-survivors)
plt.hist(survivorDF["Age"], bins = 10, color = 'g', label = 'Survivor')
plt.hist(diedDF["Age"], bins = 10, color = 'b', label = 'Died')
plt.gca().set(title = "Ages in the Titanic Dataset", ylabel = "Count", xlabel = "Ages (years)")

plt.legend()
plt.show()

# Overlapping Histograms (Age, surviors and non-surivors) transparent
plt.hist(survivorDF["Age"], bins = 10, color = 'g', label = 'Survivor', alpha = 0.5, density = True)
plt.hist(diedDF["Age"], bins = 10, color = 'b', label = 'Died', alpha = 0.5, density = True)
plt.gca().set(title = "Ages in the Titanic Dataset", xlabel = "Ages (years)", ylabel = "Density")

plt.legend()
plt.show()

# Joy plot in Python

combinedAges = [list(survivorDF["Age"]), list(diedDF["Age"])]

fig, axes = joypy.joyplot(combinedAges, labels = ["Survivors", "Casualties"], colormap = plt.cm.autumn_r)

plt.gca().set(title = "Ages in the Titanic Dataset", xlabel = "Age (years)" )
plt.show()

# Working with Seaborn, Making a box+Swarm plot

sns.boxplot(y = "Age", data = survivorDF, color = 'g')
sns.stripplot(y = "Age", data = survivorDF, color = 'k')

plt.gca().set(title = "Age of Titanic Survivors", ylabel = "Age (years)")
plt.show()

# Seaborn Violin Plot

sns.violinplot(combinedAges)
plt.gca().set(title = "Age of Titanic Survivors", ylabel = "Age (years)")
plt.show()

# Saving plot axes as a varible 
ax = sns.violinplot(combinedAges)
# Setting text of X axis Lables 
ax.set_xticklabels(["Survivors", "Casualties"])
plt.gca().set(title = "Age of Titanic Survivors", ylabel = "Age (years)")
plt.show()

# Linear Regression Confidence intervals 

#remmove missing values
titanic_Df_no_NA = titanicDF[pd.notna(titanicDF["Age"])]
titanic_Df_no_NA = titanic_Df_no_NA[pd.notna(titanic_Df_no_NA["Fare"])]

#Sort the dataframe by age
titanic_Df_no_NA = titanic_Df_no_NA.sort_values("Age")

#Add a column of 1s array. instructs statsmodel that there is a y inter
X = sm.add_constant(titanic_Df_no_NA["Age"].values)

#Ordinary least squares model given dependent (y) and independent (x) variable
ols_model = sm.OLS(titanic_Df_no_NA["Fare"].values, X)

#fits a linear model
est = ols_model.fit()
print(est.summary())

#Confidence intervak of fit parameter 
out = est.conf_int(alpha = 0.05, cols = None)

#Plotting Linear Regression Confidence Interval

#allows multiple things on the same figure 
fig, ax = plt.subplots()
titanic_Df_no_NA.plot(x = "Age", y = "Fare", marker = 's', linestyle = 'None', ax = ax)
y_pred = est.predict(X)
x_pred = titanic_Df_no_NA.Age.values
ax.plot(x_pred, y_pred)
pred = est.get_prediction(X).summary_frame()
ax.plot(x_pred, pred["mean_ci_lower"], linestyle = '--', color = 'blue')
ax.plot(x_pred, pred["mean_ci_upper"], linestyle = '--', color = 'blue')
ax.get_legend().remove()
ax.set(title = "Age vs. Fare for Titanic Passengers", ylabel = "Fare(1912 British Pound)", xlabel = "Age (years)")
plt.show()

# Heatmaps
vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
 "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
 "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
[2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
[1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
[0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
[0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
[1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
[0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

fig, ax = plt.subplots(figsize = (12.8, 9.6))
im = ax.imshow(harvest)
ax.set_xticks(range(len(farmers)), labels=farmers, rotation=45,
ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(vegetables)), labels=vegetables)
for i in range(len(vegetables)):
  for j in range(len(farmers)):
    text = ax.text(j, i, harvest[i, j],
        ha="center", va="center", color="w")
ax.set_title("Harvest of local farmers (in tons/year)")
plt.show()
