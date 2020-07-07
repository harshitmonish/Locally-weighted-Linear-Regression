# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:25:24 2020

@author: harshitm
"""


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd

def normalize(x):
    meanX = np.mean(x)
    varX = np.std(x)
    x = (x - meanX)/varX
    return x
"""
Normal linear regression using the analytical approach of calculating theta, i.e.
theta = inverse((X.T.X)).(X.T. Y)
"""
def unweighted_regression(x, y):
    y_pred = np.zeros(y.shape)
    x_comb =np.c_[np.ones(x.shape), x]
    theta = np.linalg.inv(np.dot(x_comb.T, x_comb)).dot(x_comb.T).dot(y)
    
    y_pred = np.dot(x_comb, theta)
    return y_pred
    
"""
Using the analytical approach of calculating the theta. i.e
theta = inverse((X.T.X)).(X.T. Y)
and using the weights formula.
"""

def weighted_regression(x, y, testX, tau):
    y_pred = np.zeros(y.shape)
    x_comb =np.c_[np.ones(x.shape), x]
    for i,xt in enumerate(testX):
        w = np.diag(np.exp(-((x_comb[:,1] - xt)**2)/(2*(tau**2))))
        theta = np.dot(np.linalg.inv(x_comb.T.dot(w).dot(x_comb)), x_comb.T.dot(w).dot(y))
        y_pred[i] = theta[1]*xt + theta[0]   
    
    return y_pred

def plot_graph(xtrain, ytrain, xtest, ytest, typeG):
    fig = plt.figure(figsize=(10, 8))
    plt.xlabel("Input X")
    plt.ylabel("Output Y")
    plt.scatter(xtrain, ytrain, s=5, color='red')
    xtrain.resize(100)
    ytrain.resize(100)
    xtest.resize(100)
    if(typeG == "unweighted"):
        plt.plot(xtrain, ytest)
        plt.title("Liner Regression, Unweighted")
        plt.savefig("UnweightedLR.jpg")
    else:
        plt.plot(xtest, ytest)
        plt.title("Locally weighted Linear Regression")
        plt.savefig("Locally_weighted_linear_regression.jpg")
        
    
    plt.show()
        

def main():
    #firstly read the data from the  file
    X_in = pd.read_csv("./lin_log/weightedX.csv", header=None).values
    Y_in = pd.read_csv("./lin_log/weightedY.csv", header=None).values
    #normalize the data
    X = normalize(X_in)
    testX = np.linspace(-2, 2, num=100)
    
    #first we predict y based on normal linear regression
    y_pred_unweighted = unweighted_regression(X, Y_in)
    
    #next we predict y based on locally weighted linear regression
    y_pred_weighted  = weighted_regression(X, Y_in, testX, 0.1)  
    
    #Now lets plot the predicted y
    plot_graph(X, Y_in, testX, y_pred_unweighted, "unweighted")
    plot_graph(X, Y_in, testX, y_pred_weighted, "weighted")
    
if __name__== "__main__":
    main()