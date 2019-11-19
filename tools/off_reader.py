import numpy as np
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

filepath = 'cube.off'

# Parse mesh from OFF file
#filepath = os.fsencode(filepath)
file = open(filepath, 'r')
first_line = file.readline().rstrip()


# handle blank and comment lines after the first line
# handle OFF characters on the same line as the vcount, fcount and ecount
# line = file.readline()
# while line.isspace() or line[0]=='#':
#    line = file.readline()

if first_line.split("OFF")[1]:
    line = first_line.split("OFF")[1]
    vcount, fcount, ecount = [int(x) for x in line.split()]
else:
    line = file.readline()
    vcount, fcount, ecount = [int(x) for x in line.split()]

print(vcount)
verts = []
X = []
Y = []
Z = []
facets = []
edges = []
i = 0;
while i < vcount:
    line = file.readline()
    if line.isspace():
        continue  # skip empty lines
    try:
        bits = [float(x) for x in line.split()]
        px = bits[0]
        py = bits[1]
        pz = bits[2]
        print(i)
    except ValueError:
        i = i + 1
        continue
    verts.append((px, py, pz))
    X.append(px)
    Y.append(py)
    Z.append(pz)
    i = i + 1

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(np.asarray(X), np.asarray(Y), np.asarray(Z), c=np.asarray(Z), cmap='Greens')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()