import cv2
import numpy as np
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.stats
import sklearn

# Open a simple image
img = cv2.imread("images/Hand_0000233.jpg")
img = cv2.resize(img, (360, 480))

cv2.imshow("Original", img)


# GMM Model for threshold calcultion




# converting from gbr to hsv color space
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# skin color range for hsv color space
HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

# converting from gbr to YCbCr color space
img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# skin color range for YCrCb color space
YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

# merge skin detection (YCbCr and hsv)
global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
global_mask = cv2.medianBlur(global_mask, 3)
global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

HSV_result = cv2.bitwise_not(HSV_mask)
YCrCb_result = cv2.bitwise_not(YCrCb_mask)
global_result = cv2.bitwise_not(global_mask)

final = cv2.bitwise_not(img, img, YCrCb_result)

# reducing noice from image by 0 the points lower than 50
final_touch_indices = final < 50
final[final_touch_indices] = 0

# calculate the average values for YCrCb chanels


img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# extract chanels from image
y, cr, cb = img[:, :, 0], img[:, :, 1], img[:, :, 2]

# removing zeros from chanel array
y = y[y != 0]
cr = cr[cr != 0]
cb = cb[cb != 0]

# extracting mean, max, min, standard deviation value for the chanel

yMean, yMax, yMin, yStd = y.mean(), y.max(), y.min(), y.std()
crMean, crMax, crMin, crStd = cr.mean(), cr.max(), cr.min(), cr.std()
cbMean, cbMax, cbMin, cbStd = cb.mean(), cb.max(), cb.min(), cb.std()




# normal distribution plot

# sns.distplot(y)
#
# plt.axvline(x=yMean, color='red')
#
# plt.show()

# show results
# cv2.imshow("Final", final)
cv2.imshow("1_HSV.jpg",HSV_result)
cv2.imshow("2_YCbCr.jpg",YCrCb_result)
# cv2.imshow("3_global_result.jpg",global_mask)
# cv2.imshow("Image.jpg",YCrCb_mask)
#cv2.imwrite("1_HSV.jpg",HSV_result)
#cv2.imwrite("2_YCbCrAMsk.jpg",YCrCb_result)
# cv2.imwrite("3_global_result.jpg",global_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
