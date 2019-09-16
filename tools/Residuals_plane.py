import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Parameters to set
mu_x = 0
variance_x = 0.25

mu_y = 0
variance_y = 0.25

# Create grid and multivariate normal
x = np.linspace(-1, 1, 500)
y = np.linspace(-1, 1, 500)
X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X;
pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
pdf = rv.pdf(pos)

# Create plane
plane = np.array((0.3, -0.5, 2))
z = (-X * plane[0] - Y * plane[1]) / plane[2]

# Make a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, pdf, cmap='viridis', linewidth=0)

ax.plot_surface(X, Y, z)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

x1 = X.ravel(order='C')
y1 = Y.ravel(order='C')
z1 = pdf.ravel(order='C')
points = np.column_stack((x1,y1,z1))

res = np.dot(points,plane)/np.linalg.norm(plane)
#res = res.reshape((len(x),len(y)))
#print(res.shape)

fig = plt.figure()
plot = plt.hexbin(x1, y1, C=res, gridsize=50, cmap=cm.jet, bins=None)
plt.axis([x1.min(), x1.max(), y1.min(), y1.max()])

# cb = plt.colorbar()
# cb.set_label('mean value')
plt.show()

data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
# print(data)