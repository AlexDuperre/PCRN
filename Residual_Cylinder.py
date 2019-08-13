import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import axes3d
from scipy.linalg import norm
from matplotlib import cm

# Parameters to set
mu_x = 0.5
variance_x = 0.2

mu_y = 0.5
variance_y = 0.2

# Create grid and multivariate normal
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X;
pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
pdf = rv.pdf(pos)

# Create cylinder : 7 params
cylinder = np.array((0.75, 0.2, 0.1,-0.55,0.2, 0.6, 0.12)) #restrict pylinder[0:3] to be between 0 and 1

## Find plane intersection for plotting
#  Normals n and points v0 for each plane

n_s = [[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]
v_s = [[0,0.5,0.5],[1,0.5,0.5],[0.5,0,0.5],[0.5,1,0.5],[0.5,0.5,0],[0.5,0.5,1]]

cyl_points = []
for n, v in zip(n_s,v_s):
    v = np.array(v)
    n = np.array(n)
    s = np.dot(n,v-cylinder[0:3])/np.dot(n,cylinder[3:6])
    pt = cylinder[0:3] + s*cylinder[3:6]
    print(s,pt)
    if (np.max(pt)<=1) & (np.min(pt)>=0):
        cyl_points.append(pt)

print('The two points in the unit cube are: ',cyl_points)

if len(cyl_points)>2:
    print('ERROR: More that 2 intersection points with unit cube')

# plot cylinder
#vector in direction of axis
v = cyl_points[0] - cyl_points[1]
R = cylinder[6]
#find magnitude of vector
mag = norm(v)
#unit vector in direction of axis
v = v / mag
#make some vector not in the same direction as v
not_v = np.array([1, 0, 0])
if (v == not_v).all():
    not_v = np.array([0, 1, 0])
#make vector perpendicular to v
n1 = np.cross(v, not_v)
#normalize n1
n1 /= norm(n1)
#make unit vector perpendicular to v and n1
n2 = np.cross(v, n1)
#surface ranges over t from 0 to length of axis and 0 to 2*pi
t = np.linspace(0, mag, 100)
theta = np.linspace(0, 2 * np.pi, 100)
#use meshgrid to make 2d arrays
t, theta = np.meshgrid(t, theta)
#generate coordinates for surface
x, y, z = [cyl_points[1][i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]

# Make a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z)
ax.plot_surface(X, Y, pdf, cmap='viridis', linewidth=0)



ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.view_init(35,-90)
plt.show(block=False)


# Calculating residual

x1 = X.ravel(order='C')
y1 = Y.ravel(order='C')
z1 = pdf.ravel(order='C')
points = np.column_stack((x1,y1,z1))

AC = points-cylinder[0:3]
AB = cylinder[3:6]

res = norm(np.cross(AC,AB), axis=1)/norm(AB, axis=0) - cylinder[6]


fig2 = plt.figure()
plot = plt.hexbin(x1, y1, C=res, gridsize=50, cmap=cm.jet, bins=None)
plt.axis([x1.min(), x1.max(), y1.min(), y1.max()])

cb = plt.colorbar()
plt.show()

# data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
# data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#print(data)