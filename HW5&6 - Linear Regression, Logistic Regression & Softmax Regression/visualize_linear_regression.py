import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from linear_regression import normalize, linear_regression, compute_score

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
    predicted = normalize(X) @ W + b

    ax1.set_xlim(X.min() - 0.5, X.max() + 0.5)
    ax1.set_ylim(y.min() - 0.5, y.max() + 0.5)
    ax1.set_title(f'epoch = {i + 1}, loss = {loss}, X @ {W} + {b}')
    ax1.scatter(X.reshape(-1), y, color='black')    
    ax1.plot(X.reshape(-1), predicted, color='red')
    
    ax2.set_xlim(-0.5, len(history))
    ax2.set_ylim(0, 3.5)
    ax2.plot(range(i), loss_list[:i])

if __name__ == '__main__':
    N = 100
    X = 4 * np.random.rand(N) #+ np.random.randint(10, size=N)
    e = np.random.rand(N)
    y = X + e #[(X[i] + e[i]) for i in range(len(X))]
    X = X.reshape(-1, 1)  
    loss_list = []
    lr = 0.05
    epochs = 100

    W, b, history = linear_regression(X, y, lr, epochs)
    #score = compute_score(W, b, X, y)

    print(normalize(X) @ W + b)

    fig = plt.figure(figsize=(8, 10))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ani = FuncAnimation(fig, animate, frames=len(history), 
                        interval=150, repeat=False)
    plt.show()
    plt.close()

    
