import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from perlin import *


def prikazi(teren):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    X = np.zeros_like(teren)
    Y = np.zeros_like(teren)
    for i in range(teren.shape[0]):
        for j in range(teren.shape[1]):
            X[i][j] = i
            Y[i][j] = j

    surf = ax.plot_surface(X, Y, teren, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()



pnf = PerlinNoiseFactory(2, unbias=False)
teren = np.zeros((50,50))
size_x = teren.shape[0]
size_y = teren.shape[1]
for i in range(size_x):
    for j in range(size_y):
        # values should be between -1 and 1
        teren[i][j] = pnf(i/size_x, j/size_y)

prikazi(teren)
