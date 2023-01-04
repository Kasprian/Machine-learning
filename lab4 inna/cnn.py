# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:46:54 2021

@author: Pjoter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow import keras

x_train = np.load("Xtrain_Classification_Part2.npy")
y_train = np.load("Ytrain_Classification_Part2.npy")
x_test = np.load("Xtest_Classification_Part2.npy")

train_x, test_x, train_y, test_y  = train_test_split(x_train, y_train, test_size=0.2)

train_x = np.reshape(train_x, (len(train_x), 50, 50, 1))
test_x = np.reshape(test_x, (len(test_x),50, 50, 1))

train_y_cat = tf.keras.utils.to_categorical(train_y, 4)
test_y_cat = tf.keras.utils.to_categorical(test_y, 4)


model = models.Sequential()
model.add(Input(shape=(50, 50, 1)))
model.add(layers.Rescaling(scale=1 / 255))          # scale 0-255 to 0-1
model.add(layers.Conv2D(12, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Dropout(0.2))
# Flatten
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(4, activation="softmax"))
#model.summary()

callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]


model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

history = model.fit(train_x, train_y_cat, epochs=20, batch_size=64,
                    validation_data=(test_x, test_y_cat), callbacks=callbacks)

# Evaluate the model

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(history.history['categorical_accuracy'], label='accuracy')
ax1.plot(history.history['val_categorical_accuracy'], label = 'val_accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (Cat)')
ax1.set_ylim([0.5, 1])
ax1.legend(loc='lower right')

ax2.plot(history.history['loss'], label='loss')
ax2.plot(history.history['val_loss'], label = 'val_loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (CXE)')
ax2.legend(loc='upper right')

# test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)
# print(f"Test Loss: {test_loss:.3f} Test Accuracy {test_acc:.3f}")

y_hat_test_cat = model.predict(test_x)
y_hat_test = np.argmax(y_hat_test_cat, axis=-1)
#test_loss, test_acc = mlp.evaluate(test_x,  test_y, verbose=2)

ConfusionMatrixDisplay.from_predictions(test_y, y_hat_test,display_labels=["Caucasian", "African", "Asian", "Indian"])

bacc = balanced_accuracy_score(test_y, y_hat_test)

print(f"BACC = {bacc:.3f}")
print("Y_hat shape:", y_hat_test.shape)