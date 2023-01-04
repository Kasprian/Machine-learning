# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 14:16:22 2021

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


X = np.load('Xtrain_Classification_Part1.npy')
y = np.load('Ytrain_Classification_Part1.npy')

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


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)

    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same",input_shape=(50, 50, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, (3, 3), strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    activation = "softmax"
    units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


if __name__ == "__main__":
    model = make_model(input_shape=(50,50,1), num_classes=2)
    #model.summary()
    #keras.utils.plot_model(model, show_shapes=True)
    epochs = 10
    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history=model.fit(
        train_x, train_y, epochs=epochs, callbacks=callbacks, validation_data=(test_x, test_y)
    )
    model.save("./model.h5")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(history.history["accuracy"], label="accuracy")
    ax1.plot(history.history["val_accuracy"], label="val_accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0.5, 1])
    ax1.legend(loc="lower right")

    ax2.plot(history.history["loss"], label="loss")
    ax2.plot(history.history["val_loss"], label="val_loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss ")
    ax2.legend(loc="upper right")
    #test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=True)

    #print(test_acc)