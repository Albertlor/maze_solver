# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import Nystroem, RBFSampler
from tqdm import tqdm


class Model:
  def __init__(self, maze_generator, STATE_SPACE, ACTION_SPACE, END_STATE, weights=None, featurizer=None, trained=0):
    self.maze_generator = maze_generator
    self.STATE_SPACE = STATE_SPACE
    self.ACTION_SPACE = ACTION_SPACE
    self.END_STATE = END_STATE
    self.ACTION2INT = {a: i for i, a in enumerate(self.ACTION_SPACE)}
    self.INT2ONEHOT = np.eye(len(self.ACTION_SPACE))

    # initialize linear model weights
    if trained == 0:
      # fit the featurizer to data
      samples = self.gather_samples()
      # self.featurizer = Nystroem()
      self.featurizer = RBFSampler()
      self.featurizer.fit(samples)
      dims = self.featurizer.n_components
      self.w = np.zeros(dims)
    if trained == 1:
      self.featurizer = featurizer
      self.w = weights

  def predict(self, s, a):
    sa = self.merge_state_action(s, a)
    x = self.featurizer.transform([sa])[0]
    return x @ self.w

  def predict_all_actions(self, s):
    return [self.predict(s, a) for a in self.ACTION_SPACE]

  def grad(self, s, a):
    sa = self.merge_state_action(s, a)
    x = self.featurizer.transform([sa])[0]
    return x
  
  def one_hot(self,k):
    return self.INT2ONEHOT[k]

  def merge_state_action(self,s, a):
    ai = self.one_hot(self.ACTION2INT[a])
    return np.concatenate((s, ai))

  def gather_samples(self, n_episodes=1000):
    samples = []
    for _ in tqdm(range(n_episodes)):
      self.maze_generator.i = 0
      self.maze_generator.j = 0
      s = self.maze_generator.get_current_state()
      while s!=self.END_STATE:
        a = np.random.choice(self.ACTION_SPACE)
        sa = self.merge_state_action(s, a)
        samples.append(sa)

        r = self.maze_generator.get_reward(s,a)
        s = self.maze_generator.get_current_state()
    return samples
  
  def get_optimal_configurations(self):
    # obtain V* and pi*
    V = {}
    greedy_policy = {}
    for s in self.STATE_SPACE:
      if s!=self.END_STATE:
        values = self.model.predict_all_actions(s)
        V[s] = np.max(values)
        greedy_policy[s] = self.ACTION_SPACE[np.argmax(values)]
      else:
        # terminal state or state we can't otherwise get to
        V[s] = 0