import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def logistic_regression(X, y, lr, epochs=100):
    """
    """
    m, n = X.shape
    W = np.random.randn(n)
    b = 0.0
    epoch = 0  
    loss_list = []
    # normalize data
    X = normalize(X)
    while True:
        h = sigmoid(X @ W + b)
        loss = - (np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))) / m
        loss_list.append(loss)
        dW = (X.T @ (h - y)) / m
        db = np.sum(h - y) / m
        W = W - lr * dW
        b = b - lr * db
        epoch += 1
        if epoch > epochs - 1:
            break
    return W, b, loss_list

def compute_score(W, b, X, y):
    X = normalize(X)
    predict = (sigmoid(X @ W + b) > 0.5)
    correct = predict == y
    return (np.sum(correct) / len(X)) * 100

def lor_visualize_loss(loss_list):
    sns.lineplot(x=range(len(loss_list)), y=loss_list)
    plt.show()

def test():
    X = np.array([
        [1., 2.],
        [3., 4.],
        [-1., -5.]
    ])
    y = np.array([1, 0, 1])
    W, b, loss_history = logistic_regression(X, y, lr=0.01)
    
    acc = computer_score(W, b, X, y)
    print(acc)


if __name__ == '__main__':
    test()