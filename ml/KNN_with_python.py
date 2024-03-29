#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__ = 'kira@-築城院 真鍳'

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collection import Counter
style.use('fivethirtyeight')

#plot1 = [1,3]
#plot2 = [2,5]
#euclideen_distance = sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)


dataset = {'k': [[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]]}

new_features = [5,7]

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            #euclideen_distance = sqrt( (features[0]-predict[0])** + (features[1]-predict[1])**2 )
            euclideen_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclideen_distance, group])

    votes = [i[i] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)
#[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_features[0], new_features[1], color=result)
#plt.show()
