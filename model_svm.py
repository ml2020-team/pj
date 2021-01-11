import numpy as np
import tensorflow as tf
import os

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from os.path import join as pjoin

from sklearn.svm import SVC

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print(tf.__version__)



x_train = np.load("./data/task1/avg/x_train.npy").reshape([1609, 12288])
y_train = np.load("./data/task1/avg/y_train.npy")
print(y_train.shape)

# 445
x_test = np.load("./data/task1/avg/x_test.npy").reshape([191, 12288])
y_test = np.load("./data/task1/avg/y_test.npy")
print(y_test.shape)



x_train = x_train * 1.0 / 127.5 - 1
x_test = x_test * 1.0 / 127.5 - 1
print("data load finish")
clf=SVC(kernel='linear',C=1E6)
clf.fit(x_train,y_train)
#predict = clf.predict(x_test)
print(clf.score(x_test,y_test))







