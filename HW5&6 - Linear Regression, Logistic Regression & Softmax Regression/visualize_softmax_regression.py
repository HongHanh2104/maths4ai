import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.datasets import make_blobs

from softmax_regression import softmax_regression

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
    for ci, w in enumerate(W.T):
        draw_area(w[0], w[1], b[ci], ax=ax1, 
                  x_min=X[:, 0].min() - 0.5, 
                  x_max=X[:, 0].max() + 0.5, 
                  y_min=X[:, 1].min() - 0.5, 
                  y_max=X[:, 1].max() + 0.5,
                  color=colors[ci])

    ax1.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax1.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax1.set_title(f'epoch = {i + 1}, loss = {loss}')
    sns.scatterplot(x=X[:, 0], y=X[:, 1], ax=ax1, hue=y, palette=colors[:3])
    
    #ax1.plot(X_, predicted, color='red')
    
    ax2.set_xlim(-0.5, len(history))
    #ax2.set_ylim(0, 2)
    ax2.plot(range(i), loss_list[:i])

def draw_area(W0, W1, b, ax, x_min, x_max, y_min, y_max, color):
    top = [y_max, y_max]
    bottom = [y_min, y_min]
    #left = [x_min, x_min]
    #right = [x_max, x_max]
    
    y_left = (-W0 * x_min - b) / W1
    y_right = (-W0 * x_max - b) / W1
    ax.plot([x_min, x_max], [y_left, y_right], color=color)
    
    if W1 > 0:
        ax.fill_between([x_min, x_max], [y_left, y_right], top, color=color, alpha=0.2)
    else:
        ax.fill_between([x_min, x_max],
                        bottom, [y_left, y_right], color=color, alpha=0.2)

if __name__ == '__main__':
    N = 500
    X, y = make_blobs(n_samples=N, centers=3, n_features=2)
    loss_list = []
    lr = 0.1
    epochs = 10000
    colors = ['red', 'blue', 'green', 'pink', 'black']

    W, b, history = softmax_regression(X, y, 3, lr, epochs)
    
    fig = plt.figure(figsize=(8, 10))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ani = FuncAnimation(fig, animate, frames=len(history), 
                        interval=100, repeat=False)
    plt.show()
    plt.close()

    
