import time
import numpy as np
from sklearn.metrics import roc_auc_score

class Arm:
  """ 
  Arm class representing OD models
  """

  def __init__(self, model, model_params={}):
      self.model = model
      self.model_params = model_params
      self.count = 0
      self.reward = 0.0

  def pull(self, X_train, X_validation, y_validation):
      self.count += 1
      # fit model
      start_time = time.time()
      self.model.fit(X_train)
      end_time = time.time()
      elapsed_time = end_time - start_time
      print("\t\tTraining time: %s seconds" % (elapsed_time))

      # get prediction on validation set
      y_validation_pred = self.model.predict(X_validation)  # outlier labels (0 or 1)
      y_validation_scores = self.model.decision_function(X_validation)  # outlier scores
      reward = np.round(roc_auc_score(y_validation, y_validation_scores), decimals=4)
      
      return reward, elapsed_time

  def update_reward(self, R, r):
      # Q(A) <- Q(A) + 1/N(A)[R - Q(A)]
      if(np.sum(R) != 0):
          self.reward = self.reward + (1/np.sum(R))(r - self.reward)
      else:
          self.reward = r
      print('\t\tUpdated reward:',self.reward)