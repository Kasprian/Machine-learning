# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:29:48 2021

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

image_size = (50, 50)

if __name__ == "__main__":
    model = keras.models.load_model("2save_at_5.h5")
    #x_test = np.load("Xtest_Classification_Part2.npy")
    x = np.load("Xtrain_Classification_Part2.npy")
    y = np.load("Ytrain_Classification_Part2.npy")
    # x_test = np.reshape(x_test, (len(x_test), 50, 50, 1))
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

    train_x = np.reshape(train_x, (len(train_x), 50, 50, 1))
    test_x = np.reshape(test_x, (len(test_x),50, 50, 1))
    predictions = model.predict(test_x)
    #plt.imshow(x_test[3],cmap='gray')
    #print(np.argmax(predictions[3],axis=-1))
 
    npscores= np.array(np.argmax(predictions,axis=-1))

    npscores = np.reshape(npscores, (len(npscores),1))
    bacc = balanced_accuracy_score(test_y, npscores)
    print(bacc)
    ConfusionMatrixDisplay.from_predictions(test_y, npscores, display_labels=["Caucasian", "African", "Asian", "Indian"])
    np.save("predictions.npy",npscores)
