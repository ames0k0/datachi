#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__ = 'kira@-築城院 真鍳

import pickle #--------------------------------------#
import numpy as np #---------------------------------#
from os import system #------------------------------#
from os.path import exists #-------------------------#
from sklearn.tree import DecisionTreeClassifier #----#
from sklearn.metrics import accuracy_score #---------#
from sklearn.neighbors import KNeighborsClassifier #-#
from sklearn.linear_model import LinearRegression #--#
from sklearn.model_selection import train_test_split #
from sklearn.model_selection import GridSearchCV #---#


data = 'test_linear.pkl'

class Base:

    def __init__(self):
        if exists(data):
            with open(data, 'rb') as pk:
                self.X, self.Y = pickle.load(pk)
            #self.X_train, self.X_test, self.y_train, \
            #        self.y_test = train_test_split(self.X, self.Y, test_size=0.2)
        else:
            system('python3 SoundConv.py test.wav test')
            system('clear')
            with open(data, 'rb') as pk:
                self.X, self.Y = pickle.load(pk)


class Grid(Base):
    """
    TREE:: accuracy_score:: 0.9821882951653944, param:: {'max_depth': 9}
    KNN::: accuracy_score:: 0.9974554707379135, param:: {'n_neighbors': 5}
    LIRE:: accuracy_score:: 0.9713917085869223, param:: {'n_jobs': -1}
    """

    def __init__(self):
        super().__init__()
        self.tr = DecisionTreeClassifier()
        self.li = LinearRegression()
        self.kn = KNeighborsClassifier()
        self.kn_param = {'n_neighbors': np.arange(1, 21)}
        self.tr_param = {'max_depth': np.arange(1, 11)}
        self.li_param = {'n_jobs': [-1]}
        self.trg()
        self.kng()
        self.lig()

    def trg(self):
        tree = GridSearchCV(self.tr, self.tr_param, cv=5)
        tree.fit(self.X, self.Y)
        print("TREE:: accuracy_score:: {}, param:: {}".format(tree.best_score_, tree.best_params_))

    def kng(self):
        knn = GridSearchCV(self.kn, self.kn_param, cv=5)
        knn.fit(self.X, self.Y)
        print("KNN::: accuracy_score:: {}, param:: {}".format(knn.best_score_, knn.best_params_))

    def lig(self):
        lin = GridSearchCV(self.li, self.li_param, cv=5)
        lin.fit(self.X, self.Y)
        print("LIRE:: accuracy_score:: {}, param:: {}".format(lin.best_score_, lin.best_params_))


def f():
    with open(data, 'rb') as ph:
        X, Y = pickle.load(ph)

    print(X[0], Y[0])
    knn = KNeighborsClassifier()
    X, Y = np.array(X), Y
    knn.fit(X, Y)
    print('pred:: ', knn.predict([[14, 56, 213, 4]]))
    with open('ne_model.pickle', 'wb') as nn:
        pickle.dump(knn, nn)

f()

if __name__ == "__main__":
    Grid()
