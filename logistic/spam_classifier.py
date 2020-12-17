#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/12/1 17:24
# @File    : spam_classifier.py
# @Software: PyCharm
"""
import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('spam.csv', delimiter=',', header=None)
y, X_train = df[0], df[1]
print(df.head())
print(y[0],X_train[0])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_train)
print(type(X))
print(X.shape)
c = X[0].todense()
print(c)



lr = linear_model.LogisticRegression()
lr.fit(X, y)


theta_0 = lr.intercept_
theta = lr.coef_
theta_1 = lr.coef_[0][0]
theta_2 = lr.coef_[0][1]

print(len(theta[0]))
print("theta_0 = ", theta_0)
print("theta_1 = ", theta_1)
print("theta_2 = ", theta_2)

textX = vectorizer.transform(['URGENT! Your mobile No. 1234 was awarded a Prize.',
                              'hey honey, whats up?'])

print(textX.todense())

predictions = lr.predict(textX)
print(predictions)