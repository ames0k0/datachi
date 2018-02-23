#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__ = 'kira@-築城院 真鍳'

import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style

style.use("fivethirtyeight")

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

# LINEARREGRESSION:: F: y = mx+b

def best_fit_slope(xs, ys):
    """::BEST FIT LINE::
    https://www.youtube.com/watch?v=SvmueyhSkgQ&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=8
    f: m = x*y - xy / (x)**2 - x**2
    """
    m = (((mean(xs) * mean(ys)) - mean(xs*xs)) / ((mean(xs)*mean(xs)) - mean(xs*xs)))
    return m

def best_fit_slope_and_intercept(xs, ys):
    """::BEST FIT LINE::
    https://www.youtube.com/watch?v=KLGfMGsgP34&index=9&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
    f: b = y - mx
    """
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) / ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    """::R SQUARED::
    https://www.youtube.com/watch?v=QUyAFokOmow&index=11&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
    f: r**2 = 1 - SEy^ / SEy
    """
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    # f: r**2 = 1 - SEy^ / SEy
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

# m = x*y - xy / (x)**2 - x**2, b = y - mx
m, b = best_fit_slope_and_intercept(xs, ys)

# y = mx+b
# from sklearn.cross_validation import LinearRegression
# model = LinearRegression()
# model.fit(X_train, y_train)
regression_line = [(m*x) + b for x in xs]

# model.predict(X_test, y_test)
predict_x = 8
predict_y = (m*predict_x)+b # y = mx+b

# r**2 = 1 - SEy^ / SEy
r_squared = coefficient_of_determination(ys, regression_line)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()
