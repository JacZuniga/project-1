#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:07:57 2025

@author: jacktorres
"""
import os
import pandas as pd
import scipy

directory_name = '/Users/jacktorres/Dropbox/Mac/Desktop/B DATA 200'

os.chdir(directory_name)

working_dir = os.getcwd()

print(working_dir)

file_name = "Titanic.csv"

titanicDF = pd.read_csv(file_name)

print(titanicDF.head())

# default: RangeIndex(start=0, stop=891, step=1"
titanicDF = titanicDF.set_index(list(titanicDF)[0])
print(titanicDF.index)

# getting first 400 passengers 
titanic400 = titanicDF.iloc[0:399,:]

print(titanic400.tail())

print(titanic400.shape)

#all vals where age and fare for not missing

age_fare = titanicDF[pd.notna(titanicDF["Age"])]

age_fare = age_fare[pd.notna(titanicDF["Cabin"])]

print(age_fare)

print(titanicDF.head(10))

print(age_fare.head(10))

#name sex and age of last 10 

vitals_last_10 = titanicDF.iloc[-10:,:]

vitals_last_10 = vitals_last_10.loc[:,'Name':'Age']
print(vitals_last_10)

# Stats in python
titanicDF['Age'].mean(skipna=True)

discstat = titanicDF['Age'].describe(percentiles = [0.1,0.9])
print(discstat)

#scipy
array1 = [1, 2,3]

scipy.stats.ttest_ind(array1, array2, equal_var=False,
                      nan='omit', alternative = 'less')


#chi square

scipy.stats.chi2_contingency(observed)





