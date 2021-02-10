import random
import numpy as np
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from models.solvers import random_, ucb, egreedy
from models.arm import Arm

class Bandit:
    """
    Bandit class
    """

    def __init__(self, K, arms, solver='random', solver_param={}):

        #parameters
        self.t = -1
        self.K = int(K)
        self.total_time = 0.0
        # solver
        self.solver = str(solver)
        self.solver_param = solver_param
        # arms
        self.arms = arms # arms
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
                index = random_(self.K, **self.solver_param)
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

        # Select parameters randomly
        print('\t\tSampling params for', self.arms[i].model.__class__.__name__)
        if self.arms[i].model.__class__.__name__ == 'LOF':
            self.arms[i].model.leaf_size = random.randint(10,100)
            self.arms[i].model.n_neighbors = random.randint(2,50)
        elif self.arms[i].model.__class__.__name__ == 'KNN':
            self.arms[i].model.leaf_size = random.randint(10,100)
            self.arms[i].model.n_neighbors = random.randint(2,50)
        elif self.arms[i].model.__class__.__name__ == 'IForest':
            self.arms[i].model.n_estimators = random.randint(10,200)
        elif self.arms[i].model.__class__.__name__ == 'CBLOF':
            self.arms[i].model.n_clusters = random.randint(3,20)
            self.arms[i].model.alpha = random.uniform(0.5,1)
            self.arms[i].model.beta = random.randint(2,10)
        elif self.arms[i].model.__class__.__name__ == 'COPOD':
            pass # param-free
        else:
            raise Exception('Invalid model')
        
        # Update model_params
        self.arms[i].model_params = self.arms[i].model.get_params()
        print('\t\tSampled model params:', self.arms[i].model_params)

        r, t = self.arms[i].pull(X_train, X_validation, y_validation)
        self.rewards[i].append(r)
        self.policy_payoff = np.append(self.policy_payoff, r)
        self.counts[i] += 1
        self.total_time += t
        print('\t\tTotal time:', self.total_time)

    def update_best_arm(self):
        # the arm with the greatest mean(r)
        index = {i: np.mean(r) for i, r in self.rewards.items()}
        self.best_arm = self.arms[max(index, key=index.get)]

    def get_policy_payoff(self):
        """ Compute the cumulative reward for the chosen policy """
        
        return np.mean(self.policy_payoff)
