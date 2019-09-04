import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pylab

pylab.ion()
import imageio

from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import axes3d
from scipy.linalg import norm
from matplotlib import cm


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def PlaneGrid(params):
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    z = (-X * params[0] - Y * params[1] - params[3]) / params[2]
    return x, y, z


def ShpereGrid(params):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = params[0] + params[3] * np.outer(np.cos(u), np.sin(v))
    y = params[1] + params[3] * np.outer(np.sin(u), np.sin(v))
    z = params[2] + params[3] * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def CyGrid(params):
    ## Find plane intersection for plotting
    #  Normals n and points v0 for each plane

    n_s = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    v_s = [[0, 0.5, 0.5], [1, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 0], [0.5, 0.5, 1]]

    cyl_points = []
    for n, v in zip(n_s, v_s):
        v = np.array(v)
        n = np.array(n)
        s = np.dot(n, v - params[0:3]) / np.dot(n, params[3:6])
        pt = params[0:3] + s * params[3:6]
        if (np.max(pt) <= 1) & (np.min(pt) >= 0):
            cyl_points.append(pt)

        # remove duplicates in case the intersection is on one of the cube segments
        cyl_points = list(map(np.asarray, set(map(tuple, cyl_points))))

    if len(cyl_points) > 2:
        print('ERROR: More that 2 intersection points with unit cube')

    # plot Cylinder
    # vector in direction of axis
    v = cyl_points[0] - cyl_points[1]
    R = params[6]
    # find magnitude of vector
    mag = norm(v)
    # unit vector in direction of axis
    v = v / mag
    # make some vector not in the same direction as v
    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    # make vector perpendicular to v
    n1 = np.cross(v, not_v)
    # normalize n1
    n1 /= norm(n1)
    # make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(0, mag, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    # generate coordinates for surface
    x, y, z = [cyl_points[1][i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    return x, y, z


########## main ###########
def main():
    fig = plt.figure()
    f = open("./planes.txt", "r")
    g = open("./spheres.txt", "r")
    h = open("./cyl.txt", "r")
    gif = []
    j = 0
    for i in range(file_len("planes.txt")):
        j += 1
        print(j)
        # Shape params at iteraion X
        plane = f.readline()[1:-2]
        print(plane)
        plane = np.fromstring(plane, sep=', ')

        sphere = g.readline()[1:-2]
        sphere = np.fromstring(sphere, sep=', ')

        cyl = h.readline()[1:-2]
        cyl = np.fromstring(cyl, sep=', ')

        # Computing z coordinates
        x_plane, y_plane, z_plane = PlaneGrid(plane)
        x_sphere, y_sphere, z_sphere = ShpereGrid(sphere)
        x_cy, y_cy, z_cyl = CyGrid(cyl)

        # plots
        ax = fig.add_subplot(311, projection='3d')
        ax.plot_surface(x_plane, y_plane, z_plane, cmap='viridis', linewidth=0)
        ax = fig.add_subplot(312, projection='3d')
        ax.plot_surface(x_sphere, y_sphere, z_sphere, cmap='viridis', linewidth=0)
        ax = fig.add_subplot(313, projection='3d')
        ax.plot_surface(x_cy, y_cy, z_cyl, cmap='viridis', linewidth=0)
        plt.pause(0.001)
        plt.show()

        # Save figure to array
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        gif.append(data)

    imageio.mimsave("./evolution.gif", gif)




if __name__ == '__main__':
    main()


