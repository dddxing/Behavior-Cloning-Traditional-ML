from base import RobotPolicy
import numpy as np
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.svm import LinearSVC

class POSBCRobot(RobotPolicy):
    
    """ Implement solution for Part 2 below """
    def __init__(self):
        self.clf = 0

    def train(self, data):
        # for key, val in data.items():
        #     print(key, val.shape)
        # print("Using dummy solution for POSBCRobot")
        self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel="linear"))

        X = data['obs']
        y = np.ravel(data['actions'])
        self.clf.fit(X, y)

        
    def get_actions(self, observations):
        Xs = observations
        return self.clf.predict(Xs)
        # return np.zeros(observations.shape[0])
