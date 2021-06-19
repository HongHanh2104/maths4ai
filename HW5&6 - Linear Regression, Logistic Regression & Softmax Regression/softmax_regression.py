import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def softmax(x):
    s = np.exp(x - np.max(x))
    for i in range(len(x)):
        s[i] /= np.sum(s[i])
    return s

def convert_one_hot(y, c):
    n = len(y)
    y_hot = np.zeros((n, c))
    y_hot[np.arange(n), y] = 1
    return y_hot

def softmax_regression(X, y, c, lr, epochs=100):
    # c: classes
    m, n = X.shape
    W = np.random.randn(n, c)
    b = np.random.randn(c)
    epoch = 0  
    loss_list = []
    # normalize data
    #X = normalize(X)
    while True:
        h = softmax(X @ W + b)
        loss = -np.mean(np.log(h[np.arange(len(y)), y]))
        
        y_hot = convert_one_hot(y, c)
        dW = (X.T @ (h - y_hot)) / m
        db = np.sum(h - y_hot) / m
        W = W - lr * dW
        b = b - lr * db
        loss_list.append(((W, b), loss))
        epoch += 1
        if epoch > epochs - 1:
            break
    return W, b, loss_list

def compute_score(W, b, X, y):
    #X = normalize(X)
    predict = np.argmax(softmax(X @ W + b), axis=1)
    return np.sum(y == predict)/len(y)