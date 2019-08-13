import numpy as np
import matplotlib.pylab as plt

im = plt.imread("./rgbd-dataset/apple/apple_1/apple_1_1_1_depthcrop.png")
im = plt.imread("./rgbd-dataset/banana/banana_1/banana_1_1_2_depthcrop.png")
# dims = im.shape
# im = im - (np.ones(dims)*im.max() - im + 0.05)**(-1/2)

print(im.max())
print(im.min())
print(im.mean())
plt.imshow(im/im.max()*255)
plt.show()
