# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 20:53:28 2021

@author: Pjoter
"""
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2)
accuracy = [0.7508,0.8395,0.8659,0.8930,0.9003,0.9227,0.9233,0.9339,0.9364,0.9505]
val_accuracy= [0.4274,0.5904,0.8478,0.8709,0.8748,0.8964,0.8903,0.8825,0.8872,0.8849]
loss = [0.5301,0.3827,0.3130,0.2684,0.2459,0.2092,0.1980,0.1765,0.1668,0.1423]
val_loss = [1.1111,0.6259,0.3471,0.3051,0.2809,0.2746,0.2768,0.2755,0.2780,0.3021]
ax1.plot(accuracy, label="accuracy")
ax1.plot(val_accuracy, label="val_accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.set_ylim([0.5, 1])
ax1.legend(loc="lower right")

ax2.plot(loss, label="loss")
ax2.plot(val_loss, label="val_loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss ")
ax2.legend(loc="upper right")