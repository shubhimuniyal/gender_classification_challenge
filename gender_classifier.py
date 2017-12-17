#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 17:37:04 2017

@author: shubhi
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier() 
# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3
clf1 = GaussianProcessClassifier()
clf2 = KNeighborsClassifier()
clf3 = SVC()
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf1 = clf1.fit(X,Y)
clf2 = clf2.fit(X,Y)
clf3 = clf3.fit(X,Y)
prediction = clf.predict([[190, 70, 43]])
prediction1 = clf1.predict([[190, 70, 43]])
prediction2 = clf2.predict([[190, 70, 43]])
prediction3 = clf3.predict([[190, 70, 43]])

# CHALLENGE compare their results and print the best one!
accuracy1 = accuracy_score(Y,clf1.predict(X))
accuracy2 = accuracy_score(Y,clf3.predict(X))
accuracy3 = accuracy_score(Y,clf3.predict(X))

if (accuracy1 > accuracy2) and (accuracy1 > accuracy3):
    print('most accurate GaussianProcessClassifier')
elif (accuracy2 > accuracy1) and (accuracy2 > accuracy3):
    print('most accurate KNeighborsClassifier')
else:
    print('most accurate SVC')

print(prediction,prediction1,prediction2,prediction3)
