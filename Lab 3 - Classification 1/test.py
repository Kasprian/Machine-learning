# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 22:49:23 2021

@author: Pjoter
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

image_size = (180, 180)

if __name__ == "__main__":
    model = keras.models.load_model("save_at_5.h5")
 #   x_test = np.load("Xtest_Classification_Part1.npy")
    X = np.load("Xtrain_Classification_Part1.npy")
    y = np.load("Ytrain_Classification_Part1.npy")
#    print(x_test.shape)
#    x_test = np.reshape(x_test, (len(x_test), 50, 50, 1))
 #   print(x_test.shape)

#    predictions = model.predict(x_test)
    #plt.imshow(x_test[1],cmap='gray')

#    np.save("predictions.npy",npscores)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

    train_x = np.reshape(train_x, (len(train_x), 50, 50, 1))
    test_x = np.reshape(test_x, (len(test_x),50, 50, 1))
    predictions = model.predict(test_x)
    score=[]
    for i in predictions:
        if i[0] > i[1]:
            score.append(0.0)
        else:
            score.append(1.0) 
    npscores= np.array(score)
    npscores = np.reshape(npscores, (len(npscores),1))


    bacc = balanced_accuracy_score(test_y, npscores)
    print(bacc)
    ConfusionMatrixDisplay.from_predictions(test_y, npscores, display_labels=["Man", "Women"])