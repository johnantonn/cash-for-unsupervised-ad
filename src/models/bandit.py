import numpy as np
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from models.solvers import random, ucb, egreedy
from models.arm import Arm

class Bandit:
    """
    Bandit class
    """

    def __init__(self, K, solver='egreedy', solver_param={}):

        #parameters
        self.t = -1
        self.K = int(K)
        self.total_time = 0.0
        # solver
        self.solver = str(solver)
        self.solver_param = solver_param
        # arms
        self.arms = np.array([Arm(LOF()), Arm(KNN())]) # arms
        self.best_arm = None # best arm
        # rewards
        self.counts = np.zeros(K, dtype=int)
        self.rewards = {i: [] for i in range(K)}
        # policy
        self.policy = np.array([]) # policy (sequence of played arms)
        self.policy_payoff = np.array([]) # policy payoff

    def select_arm(self):
        """ Select the arm to pull next """

        # play every arm at least once
        iz = np.where(self.counts == 0)[0]
        if len(iz) > 0:
            index = {i: 0.0 for i in range(self.K)}
            for j in iz:
                index[j] = 1.0
            print('\t\tInitialization:', index)
        
        # then, make decision (purely based on history)
        else:
            if self.solver == 'random':
                index = random(self.K, **self.solver_param)
            elif self.solver == 'egreedy':
                index = egreedy(self.K, self.t, self.rewards, **self.solver_param)
            elif self.solver == 'ucb':
                index = ucb(self.K, self.t, self.counts, self.rewards, **self.solver_param)
            else:
                raise Exception('Invalid solver')
        
        # decide the arm
        i = max(index, key=index.get)
        self.policy = np.append(self.policy, i)

        return i
    
    def play_arm(self, i, X_train, X_validation, y_validation):
        """ Play the selected arm """

        r, t = self.arms[i].pull(X_train, X_validation, y_validation)
        self.rewards[i].append(r)
        self.policy_payoff = np.append(self.policy_payoff, r)
        self.counts[i] += 1
        self.total_time += t

    def update_best_arm(self):
        # the arm with the greatest mean(r)
        index = {i: np.mean(r) for i, r in self.rewards.items()}
        self.best_arm = self.arms[max(index, key=index.get)]

    def get_policy_payoff(self):
        """ Compute the cumulative reward for the chosen policy """
        
        return np.mean(self.policy_payoff)
