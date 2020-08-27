import cv2
import numpy as np
import sklearn.mixture
import scipy.spatial.distance
import matplotlib.pyplot as plt
import imutils
import math



def getBackgroundLabel (means, precisions, labels):

    precisions_diag0 = np.diag(precisions[0])
    precisions_diag1 = np.diag(precisions[1])

    yCrCbHandColorRangeAvg = np.array([127.5, 157.5, 110])
    yCrCbHandColorRangeMin = np.array([0, 135, 85])
    yCrCbHandColorRangeMax = np.array([255, 180, 135])

    mahalanobisDistanceClass0Avg = scipy.spatial.distance.mahalanobis(yCrCbHandColorRangeAvg, means[0],precisions_diag0)
    mahalanobisDistanceClass1Avg = scipy.spatial.distance.mahalanobis(yCrCbHandColorRangeAvg, means[1],precisions_diag1)

    mahalanobisDistanceClass0Min = scipy.spatial.distance.mahalanobis(yCrCbHandColorRangeMin, means[0], precisions_diag0)
    mahalanobisDistanceClass1Min = scipy.spatial.distance.mahalanobis(yCrCbHandColorRangeMin, means[1], precisions_diag1)

    mahalanobisDistanceClass0Max = scipy.spatial.distance.mahalanobis(yCrCbHandColorRangeMax, means[0],precisions_diag0)
    mahalanobisDistanceClass1Max = scipy.spatial.distance.mahalanobis(yCrCbHandColorRangeMax, means[1],precisions_diag1)

    print("Distance for 0 :", mahalanobisDistanceClass0Min + mahalanobisDistanceClass0Max)
    print("Distance for 1 :", mahalanobisDistanceClass1Min + mahalanobisDistanceClass1Max)

    print("Average 0: ", mahalanobisDistanceClass0Avg)
    print("Average 1: ", mahalanobisDistanceClass1Avg)

    if (mahalanobisDistanceClass0Min + mahalanobisDistanceClass0Max < mahalanobisDistanceClass1Min + mahalanobisDistanceClass1Max): #& ((lables == 0).sum() > (lables == 1).sum()):

    # if ((lables == 0).sum() > (lables == 1).sum()):
        print(0)
        return 0
    else:
        print(1)
        return 1




#img = cv2.imread("images/Hand_0000223.jpg")


img = cv2.imread("/Users/chekumis/Desktop/Palmar/Hand_0001040.jpg")

img = cv2.resize(img, (360, 480))

# cv2.imshow("Final", img)

img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# extract chanels from image
y, cr, cb = img_YCrCb[:, :, 0], img_YCrCb[:, :, 1], img_YCrCb[:, :, 2]

#GMM

GMM = sklearn.mixture.GaussianMixture (n_components=2, covariance_type='diag', tol=0.001, reg_covar=1e-06,
                                 max_iter=100, n_init=1, init_params='random', weights_init=None,
                                 means_init=None, precisions_init=None, random_state=None, warm_start=False,
                                 verbose=0, verbose_interval=10)


#reshape image to 2d arrray
h, w, ycrcb = img_YCrCb.shape

d2_train_dataset = img_YCrCb.reshape((h*w,ycrcb))


GMM_results = GMM.fit(d2_train_dataset)

#GMM_results_bis= GMM.fit(TwoDim_dataset)

GMM_means = GMM.means_
GMM_covariance = GMM.covariances_

GMM_precision = GMM.precisions_

lables= GMM.fit_predict(d2_train_dataset)

print("Labels :", lables)

replace2d_dataset= d2_train_dataset.copy()

label = getBackgroundLabel(GMM_means, GMM_precision, lables)

replace2d_dataset[lables.astype(bool) == label] = [16,128,128]

reshaped_YCrCb = replace2d_dataset.reshape(h,w,ycrcb)

img_YCrCb_after = cv2.cvtColor(reshaped_YCrCb, cv2.COLOR_YCrCb2BGR)

# Ensure all background is black RGB = [0,0,0]

img_YCrCb_after[img_YCrCb_after == [16,16,16]] = 0

# --->cv2.imshow("after GMM", img_YCrCb_after)


# edge smoothing

img_RGB_after = img_YCrCb_after.copy()

img_blur = cv2.GaussianBlur(img_RGB_after, (5, 5), 0)
mask = np.zeros(img_RGB_after.shape, np.uint8)

gray = cv2.cvtColor(img_RGB_after, cv2.COLOR_BGR2GRAY)

# laplacian = cv2.Laplacian(gray,cv2.CV_64F)
#
# cv2.imshow("laplacian", laplacian)

thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)[1]

# thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_OTSU)[1]

# cv2.imshow("tresholded image", thresh)

inpaint = cv2.inpaint(img_RGB_after, thresh, 3, cv2.INPAINT_TELEA)

# cv2.imshow("inpaint image", inpaint)

kernel = np.ones((6,6),np.uint8)

closing = cv2.bitwise_not(thresh)

closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

closing = cv2.dilate(closing, None, iterations=2)
closing = cv2.erode(closing, None, iterations=2)


clop = np.where(closing[...,None] == 0 , [255,255,255],[0,0,0])

cv2.imshow("clop img", closing)

output = np.where(clop == np.array([0,0,0]),inpaint,img_RGB_after)

# cv2.imshow("final", output)


# cv2.imwrite("testowy254.jpg", output)


# output = np.where(closing == np.array([255,255,255]), np.array(img), img_YCrCb_after)
#
# cv2.imshow("after closing", output)

# contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# cv2.drawContours(mask, contours, -1, (255,255,255),5)

# output = np.where(closing == np.array([255,255,255]), inpaint, img_YCrCb_after)

# final = cv2.bitwise_not(inpaint, img_YCrCb_after, closing)

# output = np.where(mask == np.array([255, 255, 255]), img_blur, img_flatten)


#cut the fingers

gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (7, 7), 0)

edged = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
edged = cv2.dilate(edged, None, iterations=2)
edged = cv2.erode(edged, None, iterations=2)




# c = max(cnts, key=cv2.contourArea)
# hull = cv2.convexHull(c,returnPoints = False)
# defects = cv2.convexityDefects(c,hull)

circlefind = edged.copy()
cv2.imshow("show", circlefind)

def FindBiggestContour(image):
    imax = 0
    imaxcointour = -1
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # c = max(contours, key=cv2.contourArea)
    for i in range(len(contours)):
        itemp= cv2.contourArea(contours[i])
        if imaxcointour <itemp:
            imax = i
            imaxcointour = itemp
    return contours[imax]

    # srodek ciezkosci
moments = cv2.moments(circlefind)
hu = cv2.HuMoments(moments)
centres = []
centres.append((int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])))
# cv2.circle(output, centres[-1], 8, (0, 0, 255), -1)

momentPoint = centres[-1]



    #The biggest inscribed circle

VPResults = FindBiggestContour(circlefind)
dist = 0
maxdist = 0

for i in range(circlefind.shape[0]):
    for j in range(circlefind.shape[1]):
        dist = cv2.pointPolygonTest(VPResults,(i,j),True)
        if dist > maxdist:
            maxdist = dist
            center = (i,j)

cv2.circle(output, center, int(maxdist), (0, 255, 255), 2)



# cv2.circle(circlefind, centres[-1], 8, (0, 0, 255), -1)

# cv2.circle(output, center, 8, (0, 255, 255), -1)
# cv2.rectangle(output, tuple(c - int(maxdist) for c in center), tuple(c + int(maxdist) for c in center), (255, 255, 255), 2)

# cv2.circle(output, momentPoint, int(maxdist), (0, 0, 0), 2)

        # Calculating extreme points of the contour

cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)

# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

# cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
# cv2.circle(output, extLeft, 8, (0, 0, 255), -1)
# cv2.circle(output, extRight, 8, (0, 255, 0), -1)
# cv2.circle(output, extTop, 8, (255, 0, 0), -1)
# cv2.circle(output, extBot, 8, (255, 255, 0), -1)

# cv2.line(output,extBot,momentPoint,(255, 255, 0))

# cv2.line(output,center,momentPoint,(255, 255, 60), thickness = 6)


cv2.imshow("end", output)
# cv2.imshow("final", img_YCrCb_after)


print(center, momentPoint)


# rotate image




angleRadiansPoint = math.atan2(*center)

segmentX = abs(center[0] - momentPoint[0])
segmentY = abs(center[1] - momentPoint[1])

angleRadiansSegment = math.atan(segmentY/segmentX)

angleDegrees = math.degrees(angleRadiansSegment)

(h, w) = output.shape[:2]
(cX, cY) = (w // 2, h // 2)

M = cv2.getRotationMatrix2D((cX, cY), -angleDegrees, 1.0)

cos = np.abs(M[0, 0])
sin = np.abs(M[0, 1])

nW = int((h * sin) + (w * cos))
nH = int((h * cos) + (w * sin))

# rotatedImage = cv2.warpAffine(output, M, (nW, nH))
#
# cv2.imshow("rotated", rotatedImage)



#Find the biggest rectngele inside the shape

#!/usr/bin/env python

from collections import namedtuple
from operator import mul
from functools import reduce


    
Info = namedtuple('Info', 'start height')

def max_size(mat, value=0):
    """Find height, width of the largest rectangle containing all `value`'s."""
    it = iter(mat)
    hist = [(el==value) for el in next(it, [])]
    max_size = max_rectangle_size(hist)
    for row in it:
        hist = [(1+h) if el == value else 0 for h, el in zip(hist, row)]
        max_size = max(max_size, max_rectangle_size(hist), key=area)
    return max_size

def max_rectangle_size(histogram):
    """Find height, width of the largest rectangle that fits entirely under
    the histogram.
    """
    stack = []
    top = lambda: stack[-1]
    max_size = (0, 0) # height, width of the largest rectangle
    pos = 0 # current position in the histogram
    for pos, height in enumerate(histogram):
        start = pos # position where rectangle starts
        while True:
            if not stack or height > top().height:
                stack.append(Info(start, height)) # push
            elif stack and height < top().height:
                max_size = max(max_size, (top().height, (pos - top().start)),
                               key=area)
                start, _ = stack.pop()
                continue
            break # height == top().height goes here

    pos += 1
    for start, height in stack:
        max_size = max(max_size, (height, (pos - start)), key=area)
    return max_size

def area(size):
    return reduce(mul, size)




cv2.waitKey(0)& 0xFF== ord("q")
cv2.destroyAllWindows()



#calculating mehanalobis distance for the picture points

# labels = GMM.predict(d2_train_dataset)
#
# plt.scatter(d2_train_dataset[:, 1], d2_train_dataset[:, 2], c = labels, s= 30, cmap='viridis');
#
# plt.show()
