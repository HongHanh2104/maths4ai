import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(0)

def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def linear_regression(X, y, lr, epochs=100):
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
        h = X @ W + b
        loss = np.sum(np.square(h - y)) / (2*m)
        loss_list.append(loss)
        dW = (X.T @ (h - y)) / m
        db = np.sum(h - y) / m
        W = W - lr * dW
        b = b - lr * db
        epoch += 1
        if epoch > epochs - 1:
            break
    return W, b, loss_list

def lr_visualize_loss(loss_list):
    sns.lineplot(x=range(len(loss_list)), y=loss_list)
    plt.show()

def visualize(X, y, predict):
    #plt.scatter(X, y, color='black')
    plt.plot(X, predict)
    plt.show()
    plt.close()

def test():
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    X, y = load_boston(return_X_y=True)
    a, b = linear_regression(X, y, lr=0.01, epochs=100)
    print(a)
if __name__ == '__main__':
    test()