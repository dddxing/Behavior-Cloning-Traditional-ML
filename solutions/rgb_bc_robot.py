from telnetlib import XASCII
from base import RobotPolicy
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
# from sklearn import datasets, cluster
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

class RGBBCRobot(RobotPolicy):

    """ Implement solution for Part3 below """
    def __init__(self):
        self.clf = 0
        self.transformer = 0

    def train(self, data):
        obs = data["obs"]
        actions = data["actions"]
        X = obs.reshape((400,12288))
        y = actions
        y_flatten = np.ravel(y)

        # self.transformer = KernelPCA(n_components=24, kernel='linear')
        self.transformer = MDS(n_components=2)
        X_transformed = self.transformer.fit_transform(X)
        

        # self.clf = tree.DecisionTreeClassifier()
        # self.clf.fit(X_transformed, y_flatten)

        # self.clf = LinearDiscriminantAnalysis()
        # self.clf.fit(X_transformed, y_flatten)

        self.clf = KNeighborsClassifier(n_neighbors=3)
        self.clf.fit(X_transformed, y_flatten)


        # self.clf = SVC(gamma='auto', kernel="sigmoid",decision_function_shape='ovo')
        # self.clf.fit(X_transformed, y_flatten)

        # X, y = make_classification(n_samples=400, n_features=2,
        #                             n_informative=2, n_redundant=0,
        #                             random_state=0, shuffle=False)
        # self.clf = RandomForestClassifier(max_depth=2, random_state=0)
        # self.clf.fit(X_transformed, y_flatten)


    def get_actions(self, observations):
        # print(observations.shape)
        observations = observations.reshape((2, 6144))
        observations = self.transformer.fit_transform(observations)
        prediction = self.clf.predict(observations)
        return prediction
        
    
