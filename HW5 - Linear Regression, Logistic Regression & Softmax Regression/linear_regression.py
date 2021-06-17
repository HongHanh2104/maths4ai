import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)

def linear_regression(X, y, lr, epochs=1500):
    """
    """
    m, n = X.shape
    W = np.random.randn(n)
    b = 0
    epoch = 0  
    loss_list = []

    while True:
        if epoch > epochs - 1:
            break
        h = X @ W + b
        loss = np.sum(np.square(h - y)) / (2*m)
        loss_list.append(loss)
        #dW = (X.T @ (h - y)) / m
        dW = (X.T @ (h - y)) / m
        db = np.sum(h - y) / m
        W = W - lr * dW
        b = b - lr * db
        epoch += 1
    return X @ W + b, loss_list

def visualize_loss(loss_list):
    sns.lineplot(x=range(len(loss_list)), y=loss_list)
    plt.show()

def visualize(X, y, predict):
    #plt.scatter(X, y, color='black')
    plt.plot(X, predict)
    plt.show()
    plt.close()
