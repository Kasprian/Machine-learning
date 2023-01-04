# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 06:46:11 2021

@author: Pjoter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

x = np.load("Xtrain_Classification_Part2.npy")
y = np.load("Ytrain_Classification_Part2.npy")



#plt.imshow(np.reshape(x[2],(50,50)),cmap='gray')
#plt.imshow(y[2],cmap='gray')

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2)

clf = Pipeline([
    ("preprocessing", StandardScaler()),
    ("classifier", MLPClassifier(hidden_layer_sizes=(128,256), random_state=0,verbose= True))
])
clf.fit(x_train, y_train)

y_predicted = clf.predict(x_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_predicted,display_labels=["Caucasian", "African", "Asian", "Indian"])
bacc = balanced_accuracy_score(y_test, y_predicted)
print(f"BACC = {bacc}")