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



class MachineLearningThresholding():

	def __init__(self, data_I, data_Q, target_classifier, auto=True, label='', timestamp=None,
                 fig_format='png',
                 q0_label='q0',
                 q1_label='q1', close_fig=True, **kw):


	self.X = np.zeros(len(data_I),2)
	self.X[:,0] = data_I
	self.X[:,1] = data_Q
	self.y = training_classifier

	def train_classifier_on_calibration_point(self):
		name = 'SVC'
		clf = SVC(gamma=2, C=1)

		figure = plt.figure(figsize=(27, 9))
		i = 1
		# preprocess dataset, split into training and test part
		self.X = StandardScaler().fit_transform(self.X)
		self.X_train, self.X_test, self.y_train, self.y_test = \
		    train_test_split(self.X, self.y, test_size=.4, random_state=42)

		self.x_min, self.x_max = self.X[:, 0].min() - .5, self.X[:, 0].max() + .5
		self.y_min, self.y_max = self.X[:, 1].min() - .5, self.X[:, 1].max() + .5
		self.xx, self.yy = np.meshgrid(np.arange(self.x_min, self.x_max, h),
		                     np.arange(self.y_min, self.y_max, h))

		# just plot the dataset first
		cm = plt.cm.RdBu
		cm_bright = ListedColormap(['#FF0000', '#0000FF'])
		ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
		ax.set_title("Input data")
		# Plot the training points
		ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=cm_bright,
		           edgecolors='k')
		# and testing points
		ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=cm_bright, alpha=0.6,
		           edgecolors='k')
		ax.set_xlim(self.xx.min(), self.xx.max())
		ax.set_ylim(self.yy.min(), self.yy.max())
		ax.set_xticks(())
		ax.set_yticks(())
		i += 1
		ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
		clf.fit(X_train, y_train)
		score = clf.score(X_test, y_test)

		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, x_max]x[y_min, y_max].
		if hasattr(clf, "decision_function"):
		    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
		else:
		    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

		# Put the result into a color plot
		Z = Z.reshape(xx.shape)
		ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

		# Plot also the training points
		ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
		           edgecolors='k')
		# and testing points
		ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
		           edgecolors='k', alpha=0.6)

		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		ax.set_xticks(())
		ax.set_yticks(())
		ax.set_title(name)
		ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
		        size=15, horizontalalignment='right')
		i += 1

		# plt.tight_layout()
		# plt.show()


	# plt.tight_layout()
	# plt.show()


