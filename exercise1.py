# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:27:16 2020

@author: Mathias
"""

import os
import tarfile
import urllib
import pandas as pd
import numpy as np

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#%%

# a)
    
import matplotlib.pyplot as plt
from scipy.stats import norm

fetch_housing_data()
housing = load_housing_data()
    
median_income = housing["median_income"]
    
bins = 50
x = np.linspace(np.min(median_income), np.max(median_income), bins)
plt.hist(median_income, bins, density=True)
plt.xlabel('Median income')
plt.ylabel('Frequency')

mean = median_income.mean()
median = median_income.median()
std = median_income.std()

print('mean: ' + str(mean))
print('median: ' + str(median))
print('std.dev.: ' + str(std))

#%%

# b) forskel på median og middelværdi? Hvilken beskriver bedst?

# median beskriver bedst og er mindre følsom overfor outliers

#%%

# c) Fit en normalfordeling til data og plot histogrammet – passer de to ? 

bins = 50
x = np.linspace(np.min(median_income), np.max(median_income), bins)
plt.hist(median_income, bins, density=True)
plt.plot(x, norm.pdf(x, mean, std))
plt.xlabel('Median income')
plt.ylabel('Frequency')
plt.show()

#%%

# d) lav korrelationsplot

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

#%%

# e) Hvad er 5% og 95% percentilerne af median_house_value ? (dvs. grænserne for 5% laveste og højeste). 
# Plot også fordelingen af median_house_value. Kommentér på realismen af max-værdi og 95% percentil
# Foreslå gerne en løsning til hvad man kan gøre ved dette, hvis man skal have mere realistiske data

median_house_value = housing["median_house_value"]

# Finding percentiles
p5 = np.percentile(median_house_value, 5)
p95 = np.percentile(median_house_value, 95)

print("5th percentile: " + str(p5))
print("95th percentile: " + str(p95))

# Plotting distribution
plt.hist(median_house_value, bins=100)
plt.xlabel('Median house value')
plt.ylabel('Frequency')
plt.show()







