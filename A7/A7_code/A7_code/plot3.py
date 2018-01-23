import os
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
filename = 'part3.npy'

x=np.linspace(-1.2,0.6,50)
y=np.linspace(-0.07,0.07,50)
X,Y = np.meshgrid(x,y)

if os.path.exists(filename):

    data = np.load(filename)

    ax.plot_wireframe(X,Y,data)

    plt.ylim([-0.07,0.07])
    plt.xlim([-1.2,0.6])
    plt.xlabel('position')
    plt.ylabel('velocity')
    plt.legend()
    plt.show()