from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

img = cv2.imread("images/Hand_0000282.jpg")
img = cv2.resize(img, (360, 480))

img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


y, cr, cb = img_YCrCb[:, :, 0], img_YCrCb[:, :, 1], img_YCrCb[:, :, 2]

x = y
y = cr
z = cb



ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('Y Label')
ax.set_ylabel('Cr Label')
ax.set_zlabel('Cb Label')

plt.show()