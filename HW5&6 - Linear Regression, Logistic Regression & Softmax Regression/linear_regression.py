import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import seaborn as sns
import os

np.random.seed(0)

def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def linear_regression(X, y, lr, epochs=100):
    m, n = X.shape
    W = np.random.randn(n)
    b = 0.0
    epoch = 0  
    history = []
    # normalize data
    X = normalize(X)
    while True:
        h = X @ W + b
        loss = np.sum(np.square(h - y)) / (2*m)
        dW = (X.T @ (h - y)) / m
        db = np.sum(h - y) / m
        W = W - lr * dW
        b = b - lr * db
        history.append(((W, b), loss))
        epoch += 1
        if epoch > epochs - 1:
            break
    return W, b, history

def compute_score(W, b, X, y):
    #X = normalize(X)
    predict = normalize(X) @ W + b 
    return (1 - (np.sum(((y - predict)**2))/np.sum((y - np.mean(y))**2)))

