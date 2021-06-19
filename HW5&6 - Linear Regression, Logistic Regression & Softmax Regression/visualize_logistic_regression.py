import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.datasets import make_blobs

from logistic_regression import logistic_regression

plt.style.use('seaborn-pastel')
plt.rcParams.update({'font.size': 8})

def animate(i):
    ax1.clear()
    ax2.clear()

    data = history[i]
    W = data[0][0]
    b = data[0][1]
    loss = data[1]
    loss_list.append(loss)
    X_ = np.array([X.min() - 1, X.max() + 1])
    predicted = - (b + np.dot(W[0], X_)) / W[1]

    ax1.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax1.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax1.set_title(f'epoch = {i + 1}, loss = {loss}')
    #ax1.scatter(X.reshape(-1), y, color='black')    
    sns.scatterplot(x=X[:, 0], y=X[:, 1], ax=ax1, hue=y)
    ax1.plot(X_, predicted, color='red')
    
    ax2.set_xlim(-0.5, len(history))
    ax2.set_ylim(0, 2)
    ax2.plot(range(i), loss_list[:i])

if __name__ == '__main__':
    N = 1000
    X, y = make_blobs(n_samples=N, centers=2, n_features=2)
    loss_list = []
    lr = 0.05
    epochs = 1000

    W, b, history = logistic_regression(X, y, lr, epochs)
    
    fig = plt.figure(figsize=(8, 10))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ani = FuncAnimation(fig, animate, frames=len(history), 
                        interval=100, repeat=False)
    plt.show()
    plt.close()

    
