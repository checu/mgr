import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('images/Hand_0000345.jpg')
img = cv2.resize(image, (360,480))

mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (50, 50, 350, 680)

# cv2.rectangle(img, (10, 20), (650, 320), (0,255,0), 2)
# cv2.imshow("Video Feed", img)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img),plt.colorbar(),plt.show()