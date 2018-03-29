import time
import numpy as np
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
# import dataprep for tomography module
# import tomography module
# using the data prep module of analysis V2
# from pycqed.analysis_v2 import tomography_dataprep as dataprep
from pycqed.analysis import measurement_analysis as ma
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
try:
    import qutip as qt
except ImportError as e:
    pass
    # logging.warning('Could not import qutip, tomo code will not work')



class MachineLearningThresholding(object):

    def __init__(self, X_calibration, target_classifier, mesh_step_size, **kw):


        self.X = X_calibration
        self.y = target_classifier
        self.h = mesh_step_size  # step size in the mesh

        self.names = ["RBF SVM"]

        self.clf = SVC(gamma=2, C=1)
    def train_classifier_on_calibration_point(self, plot=False):
        if plot:
            figure = plt.figure(figsize=(8, 8))
        i = 1
        # preprocess dataset, split into training and test part
        self.X = StandardScaler().fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(self.X, self.y, test_size=.4, random_state=42)

        self.x_min, self.x_max = self.X[:, 0].min() - .5, self.X[:, 0].max() + 0.5
        self.y_min, self.y_max = self.X[:, 1].min() - .5, self.X[:, 1].max() + 0.5
        self.xx, self.yy = np.meshgrid(np.arange(self.x_min, self.x_max, self.h),
                np.arange(self.y_min, self.y_max, self.h))

        if plot:
            # just plot the dataset first
            cm = plt.cm.RdBu
            cm_bright = ListedColormap(['#FF0000', '#0000FF'])
            #len(datasets) hardcoded to be 3
            ax = plt.subplot(3, 2, 1)
            ax.set_title("Input data")
            # Plot the training points
            ax.scatter(self.X_train[:, 0],
                       self.X_train[:, 1],
                       c=self.y_train,
                       cmap=cm_bright,
                       edgecolors='k')
            # and testing points
            ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=cm_bright, alpha=0.6,
                edgecolors='k')
            ax.set_xlim(self.xx.min(), self.xx.max())
            ax.set_ylim(self.yy.min(), self.yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            # iterate over classifiers
            
            ax = plt.subplot(3, 2, 1)
        self.clf.fit(self.X_train, self.y_train)
        score = self.clf.score(self.X_test, self.y_test)

        if plot:

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(self.clf, "decision_function"):
                self.Z = self.clf.decision_function(np.c_[self.xx.ravel(), self.yy.ravel()])
            else:
                self.Z = self.clf.predict_proba(np.c_[self.xx.ravel(), self.yy.ravel()])[:, 1]

            # Put the result into a color plot
            self.Z = self.Z.reshape(self.xx.shape)
            ax.contourf(self.xx, self.yy, self.Z, cmap=cm, alpha=.8)

            # Plot also the training points
            ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=cm_bright,
                       edgecolors='k')
            # and testing points
            ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=cm_bright,
                       edgecolors='k', alpha=0.6)

            ax.set_xlim(self.xx.min(), self.xx.max())
            ax.set_ylim(self.yy.min(), self.yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title('RBF SVM')
            ax.text(self.xx.max() - .3, self.yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            

        return self.clf, score




