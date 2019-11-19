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
pos[:, :, 0] = X
pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
pdf = rv.pdf(pos)

# Create sphere : 4 params
sphere = np.array((0.25, -0.75, 0.45, 0.25))

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = sphere[0] + sphere[3] * np.outer(np.cos(u), np.sin(v))
y = sphere[1] + sphere[3] * np.outer(np.sin(u), np.sin(v))
z = sphere[2] + sphere[3] * np.outer(np.ones(np.size(u)), np.cos(v))

# Make a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, pdf, cmap='viridis', linewidth=0)

ax.plot_surface(x, y, z)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()


# Residuals
x1 = X.ravel(order='C')
y1 = Y.ravel(order='C')
z1 = pdf.ravel(order='C')
points = np.column_stack((x1,y1,z1))

res = np.linalg.norm(points-sphere[0:3],axis=1) - sphere[3]

fig = plt.figure()
plot = plt.hexbin(x1, y1, C=res, gridsize=50, cmap=cm.jet, bins=None)
plt.axis([x1.min(), x1.max(), y1.min(), y1.max()])

cb = plt.colorbar()
# cb.set_label('mean value')
plt.show()

# data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
# data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
# print(data)