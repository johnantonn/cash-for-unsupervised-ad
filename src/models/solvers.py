""" 
Solver functions for selecting bandit arms 
"""

import math, random
import numpy as np

def random_(K):
    """
    K = the number of arms (domains)
    """
    print('\t\tRandom selection')
    index = {i: random.random() for i in range(K)}
    return index

def ucb(K, t, counts, R):
    """
    K = the number of arms (domains)
    t = current timestep
    counts = the total number of trials
    R = the sequence of past rewards
    """
    
    index = {}
    for i, r in R.items():
        n_i = counts[i]
        mu_i = np.mean(r)
        # At = Qt(A) + c*sqrt(lnt/Nt(a))
        # bound = np.sqrt((2 * np.log(n_i)) / n_i)
        bound = np.sqrt((2 * np.log(t)) / n_i)
        index[i] = mu_i + bound
        
    return index

def egreedy(K, t, R, epsilon=0.1, decay=True):
    """
    K = the number of arms (domains)
    t = current timestep
    R = the sequence of past rewards
    epsilon = the epsilon constant
    decay = decreasing epsilon over time
    """
    
    if decay:
        epsilon = 1.0 / np.log(t + 0.00001)
    
    if np.random.uniform(0.0, 1.0) > epsilon:
        # exploitation
        index = {i: np.mean(r) for i, r in R.items()}
        print('\t\tExploitation:',index)
    else:
        # exploration
        index = {i: np.random.uniform(0.0, 1.0) for i in range(K)}
        print('\t\tExploration:',index)
        
    return index

def softmax(K, R):
    """
    K = the number of arms (domains)
    R = the sequence of past rewards
    """
    
    softmax = np.zeros(K, dtype=float)
    for i, r in R.items():
        softmax[i] = np.mean(r)
        
    softmax = np.exp(softmax) / np.exp(softmax).sum()
    si = np.random.choice(np.arange(0, K, 1), size=1, p=softmax)[0]
    index = {i: 0.0 for i in range(K)}
    index[si] = 1.0
    
    return index