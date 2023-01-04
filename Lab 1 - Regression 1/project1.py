# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 12:09:38 2021

@author: Pjoter
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 18:47:40 2021

@author: Pjoter
"""
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import balanced_accuracy_score as BACC
from numpy import mean
from numpy import std

def cross(X, y, model):
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    print(model)
    print(scores)
    print('M: %.3f (%.3f)' % (mean(scores), std(scores)))
    
def scores(y_real,y_pred,mode):
    ###y_real - ground truth vector 
    ###y_pred - vector of predictions, must have the same shape as y_real
    ###mode   - if evaluating regression ('r') or classification ('c')
    
    if y_real.shape != y_pred.shape:
        print('confirm that both of your inputs have the same shape')
    else:
        if mode == 'r':
            mse = MSE(y_real,y_pred)
            print('The Mean Square Error is', mse)
            return mse
        
        elif mode == 'c':
            bacc = BACC(y_real,y_pred)
            print('The Balanced Accuracy is', bacc)
            return bacc
        
        else:
            print('You must define the mode input.')


def main():
    X = np.load('Xtrain_Regression_Part1.npy')
    y = np.load('Ytrain_Regression_Part1.npy')
    X_val = np.load('Xtest_Regression_Part1.npy')
    cross(X, y, LinearRegression())
    cross(X, y, Lasso())
    cross(X, y, Ridge())
    # Best result obtained by Ridghe method
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # summarize the shape of the training dataset
    print(X_train.shape, y_train.shape)

    cross(X_train, y_train, LinearRegression())
    cross(X_train, y_train, Lasso())
    cross(X_train, y_train, Ridge())
    #Best result once again with Ridge

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_val = model.predict(X_test)
    print(MSE(y_val,y_test))
    #print(y_val.shape)
    #np.save("predictions.npy",y_val)
    
if __name__ == "__main__":
    main()