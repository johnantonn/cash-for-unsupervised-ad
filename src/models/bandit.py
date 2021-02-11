import random
import numpy as np
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from models.solvers import random_, ucb, egreedy
from models.arm import Arm
from multiprocessing import Process, Manager
import time

class Bandit:
    """
    Bandit class
    """

    def __init__(self, K, arms, solver='random', solver_param={}):

        #parameters
        self.t = -1 # maybe: total_time / timeout ?
        self.K = int(K)
        self.total_time = 0.0
        # solver
        self.solver = str(solver)
        self.solver_param = solver_param
        # arms
        self.arms = arms # arms
        self.best_arm = None # best arm
        self.best_params = {}
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
    
    def play_arm(self, i, X_train, X_validation, y_validation, timeout):
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
            self.arms[i].model.n_clusters = random.randint(5,20)
            self.arms[i].model.alpha = random.uniform(0.5,1)
            self.arms[i].model.beta = random.randint(2,10)
        elif self.arms[i].model.__class__.__name__ == 'COPOD':
            pass # param-free
        else:
            raise Exception('Invalid model')
        
        # Update model_params
        print('\t\tSampled model params:', self.arms[i].model.get_params())
        self.arms[i].model_params.append(self.arms[i].model.get_params())

        # pull arm
        start = time.time()
        manager = Manager()
        return_dict = manager.dict()
        p = Process(target=self.arms[i].pull, args=(X_train, X_validation, y_validation, return_dict))
        p.start()
        p.join(timeout)
        p.terminate()
        end = time.time()
        print('\t\tPull arm elapsed time:', end-start)

        if p.exitcode == 0:
            r = return_dict['reward']
            t = return_dict['elapsed_time']
        elif p.exitcode is None:
            r = 0.0
            t = timeout

        self.rewards[i].append(r)
        self.policy_payoff = np.append(self.policy_payoff, r)
        self.counts[i] += 1
        self.total_time += t
        print('\t\tTotal time:', self.total_time)

    def update_best_arm(self):
        # the arm with the greatest mean(r)
        index = {i: np.mean(r) for i, r in self.rewards.items()} # by best mean reward across arms
        best_arm_index = max(index, key=index.get)
        self.best_arm = self.arms[best_arm_index]
        # the params that yielded the best score
        best_params_index = self.rewards[best_arm_index].index(max(self.rewards[best_arm_index])) # by best reward in the best arm
        self.best_params = self.arms[best_arm_index].model_params[best_params_index]

    def get_policy_payoff(self):
        """ Compute the cumulative reward for the chosen policy """
        
        return np.mean(self.policy_payoff)
