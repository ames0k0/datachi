#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__ = 'kira@-築城院 真鍳

import pickle #------------------------------------#
from os import system #----------------------------#
from os.path import exists #-----------------------#
from sklearn.neighbors import KNeighborsClassifier #


sc = '../model.pickle'
data = 'test_linear.pkl'

def load():
    if exists(sc):
        with open(sc, 'rb') as md:
            return pickle.load(md)
    return KNeighborsClassifier()
    # TO CHANGE MODEL::
    # python3 linear_test.py
    # DATA:: linear_test.pkl -> train data

def dump(fited):
    with open(sc, 'wb') as md:
        pickle.dump(fited, md)

"""
if c == 3:
    if n < 40: upd(4)
    elif n > 40 and n < 140: upd(0)
    elif n > 140 and n < 180: upd(1)
    elif n > 180 and n < 220: upd(3)
    elif n > 220: upd(2)
"""

def sv(x, y):
    with open(data, 'wb') as pk:
        pickle.dump((x, y), pk)
