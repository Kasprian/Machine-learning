# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:07:05 2021

@author: Pjoter
"""


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

X = np.load("Xtrain_Classification_Part2.npy")
y = np.load("Ytrain_Classification_Part2.npy")
x_test = np.load("Xtest_Classification_Part2.npy")

X=X/255

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier()

clf.fit(x_train,y_train)
#y_hat_train = clf.predict(x_train)
y_hat_test = clf.predict(x_test)
#print(y_hat_test[1])
#plt.imshow(np.reshape(x_test[1],(50, 50, 1)),cmap='gray')

bacc_val  = balanced_accuracy_score(y_test, y_hat_test)
print("BACC:", bacc_val)

ConfusionMatrixDisplay.from_predictions(y_test, y_hat_test,display_labels=["Caucasian", "African", "Asian", "Indian"])
#disp.plot()
#plt.show()