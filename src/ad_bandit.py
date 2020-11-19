# Random Selection

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('datasets/Ads_Optimisation.csv')

# Implementing Random Selection
import random
N = 10000
d = 10
selected_actions = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    selected_actions.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

# Total reward
print("Total sum of rewards:", total_reward)

# Summarize counts of selected actions
print(pd.Series(selected_actions).tail(1000).value_counts(normalize = True))