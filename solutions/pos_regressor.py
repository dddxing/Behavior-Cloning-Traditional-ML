from base import Regressor
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, ElasticNetCV, BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV, cross_validate, train_test_split

class PositionRegressor(Regressor):

    """ Implement solution for Part 1 below  """
    def __init__(self):
        self.clf = 0

    def train(self, data):
        # for key, val in data.items():
        #     print(key, val)
        # print("Using dummy solution for PositionRegressor")
        obs = data["obs"]
        info = data["info"]
        pos = np.array([a["agent_pos"] for a in info])
        X = obs.reshape((500, 12288))
        y = pos
        self.clf = Ridge(alpha=1.0)
        self.clf.fit(X, y)

    def predict(self, Xs):
        n = len(Xs)
        Xs = Xs.reshape((n, 12288))
        return self.clf.predict(Xs)
        