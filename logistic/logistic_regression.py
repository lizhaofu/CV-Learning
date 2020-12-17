#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/11/26 11:15
# @File    : logistic_regression.py
# @Software: PyCharm
"""
import joblib
from sklearn import linear_model

import numpy as np

a = np.array([1,2,3,4])
print(a)



print("ok")

X = [[20, 3],
     [23, 7],
     [31, 10],
     [42, 13],
     [50, 7],
     [60, 5]]

y = [0,
     1,
     1,
     1,
     0,
     0]

lr = linear_model.LogisticRegression()

lr.fit(X, y)

testX = [[28, 8]]

label = lr.predict(testX)

print('predicted label = ', label)

prob = lr.predict_proba(testX)

print('probability = ', prob)

print('log probability = ', lr.predict_log_proba(testX))

print(lr.get_params())
# print(lr.score(testX, y=None))



joblib.dump(lr, 'save/lr-model-new.m')

print('ok2')

theta_0 = lr.intercept_
theta = lr.coef_
theta_1 = lr.coef_[0][0]
theta_2 = lr.coef_[0][1]

print("theta_0 = ", theta_0)
print("theta_1 = ", theta_1)
print("theta_2 = ", theta_2)


testX = [[28, 8]]
ratio = prob[0][1] / prob[0][0]

testXM = [[28, 9]]
prob_new = lr.predict_proba(testXM)
ratio_new = prob_new[0][1] / prob_new[0][0]

ratio_of_ratio = ratio_new / ratio
print('ratio_of_ratio = ', ratio_of_ratio)

import math

theta_2_e = math.exp(theta_2)
print("theta_2_e = ", theta_2_e)



