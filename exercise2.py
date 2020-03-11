# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:21:12 2020

@author: Mathias
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

GAME_DATA_PATH = os.path.join("./datasets/game_data/")

def load_game_data(game_data_path=GAME_DATA_PATH):
    csv_path = os.path.join(game_data_path, "dump2.csv")
    return pd.read_csv(csv_path, sep=";")

#%%

game_data = load_game_data()
playerId = game_data["playerId"]

print(game_data.info())
print('Number of unique players: ' + str(len(playerId.unique())))

#game_data.hist(bins, figsize=(20, 15))
#game_data.plot(kind="scatter", x="playerId", y="totalNumberOfCorrectGuesses")

#%%

totalNumberOfSongsPlayed = game_data["totalNumberOfSongsPlayed"]
totalNumberOfCorrectGuesses = game_data["totalNumberOfCorrectGuesses"]
level = game_data["level"]

game_data.info()

bins = 50

plt.figure(1)
plt.hist(totalNumberOfSongsPlayed, bins)
plt.xlabel('Total number of songs played')
plt.ylabel('Frequency')

plt.figure(2)
plt.hist(totalNumberOfCorrectGuesses, bins)
plt.xlabel('Total number of correct guesses')
plt.ylabel('Frequency')

plt.figure(3)
plt.hist(level, bins)
plt.xlabel('Level')
plt.ylabel('Frequency')

#%%

unique_game_data = game_data.drop_duplicates(subset=["playerId"], keep='last')
level = unique_game_data["level"]

plt.figure(3)
plt.hist(level, bins)
plt.xlabel('Level')
plt.ylabel('Frequency')

mean = level.mean()
median = level.median()
std = level.std()

print('mean: ' + str(mean))
print('median: ' + str(median))
print('std.dev.: ' + str(std))

#%%
totalNumberOfCorrectGuesses = unique_game_data["totalNumberOfCorrectGuesses"]

plt.figure(5)
unique_game_data.plot(kind="scatter", x="playerId", y="totalNumberOfCorrectGuesses")

#%%
plt.figure(7)
bins = 30
test = unique_game_data["totalNumberOfCorrectGuesses"] / unique_game_data["totalNumberOfSongsPlayed"] * 100
x = np.linspace(np.min(test), np.max(test), bins)
mean = test.mean()
std = test.std()
plt.hist(test, bins, density=True)
plt.plot(x, norm.pdf(x, mean, std))
plt.xlabel('Correct guess percentage')
plt.ylabel('Density')
plt.show()

#%%

from pandas.plotting import scatter_matrix

attributes = ["totalNumberOfSongsPlayed", "totalNumberOfCorrectGuesses", "level"]
scatter_matrix(unique_game_data[attributes], figsize=(12,8))

#%%

unique_game_data["correct_guess_percentage"] = unique_game_data["totalNumberOfCorrectGuesses"] / unique_game_data["totalNumberOfSongsPlayed"] * 100

corr_matrix = unique_game_data.corr()
print(corr_matrix["correct_guess_percentage"].sort_values(ascending=False))

correct_guess_percentage = unique_game_data["correct_guess_percentage"]

mean = correct_guess_percentage.mean()
median = correct_guess_percentage.median()
std = correct_guess_percentage.std()
p95 = np.percentile(correct_guess_percentage, 95)

print('mean: ' + str(mean))
print('median: ' + str(median))
print('std.dev.: ' + str(std))
print("95th percentile: " + str(p95))

