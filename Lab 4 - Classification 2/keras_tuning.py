# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:07:58 2021

@author: Pjoter
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import tensorflow as tf
import pydot
import graphviz
from sklearn.model_selection import train_test_split


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

X = np.load('Xtrain_Classification_Part2.npy')
y = np.load('Ytrain_Classification_Part2.npy')

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

train_x = np.reshape(train_x, (len(train_x), 50, 50, 1))
test_x = np.reshape(test_x, (len(test_x),50, 50, 1))

#plt.imshow(train_x[2],cmap='gray')
#print(y[2])
    
batch_size = 32

# model

num_classes = 2

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
    ]
)

def build_model(hp):
    # create model object
    model = keras.Sequential([
    #adding first convolutional layer    
    keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        #activation function
        activation='relu',
        input_shape=(50,50,1)),
    # adding second convolutional layer 
    keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        #adding filter size or kernel size
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        #activation function
        activation='relu'
    ),
    # adding flatten layer    
    keras.layers.Flatten(),
    # adding dense layer    
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ),
    # output layer    
    keras.layers.Dense(10, activation='softmax')
    ])
    #compilation of model
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model


if __name__ == "__main__":
    model = make_model(input_shape=(50,50,1), num_classes=4)
    #model.summary()
    #keras.utils.plot_model(model, show_shapes=True)
    epochs = 15
    callbacks = [
        keras.callbacks.ModelCheckpoint("tuning_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_x, train_y, epochs=epochs, callbacks=callbacks, validation_data=(test_x, test_y)
    )
    model.save("./model.h5")
    #test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=True)

    #print(test_acc)