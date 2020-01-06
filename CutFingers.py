import cv2
import numpy as np
import sklearn.mixture
import scipy.spatial.distance
import argparse
import imutils
from imutils import contours
from skimage import measure

def conectionPoints(output):

    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=2)

    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c,returnPoints = False)
    defects = cv2.convexityDefects(c,hull)

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(c[s][0])
        end = tuple(c[e][0])
        far = tuple(c[f][0])

        # cv2.line(img,start,end,[0,255,0],2)precisions_diag1 = precisions[0].diag(precisions)
        cv2.circle(img,far,5,[0,0,255],-1)


# find the darkest spots

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to the image file")
# args = vars(ap.parse_args())


img = cv2.imread("testowy106.jpg")

conectionPoints(img)

cv2.imshow("pointsDetection", img)

(h, w) = img.shape[:2]

# cv2.imshow("original", img)
img[img == 0] = 255
# cv2.imshow("original", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
thresh = cv2.threshold(blurred, 95, 255, cv2.THRESH_BINARY_INV)[1]

# cv2.imshow("pointsDetection", thresh)

thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)



# delete elements near the border
# cv2.rectangle(thresh, (0,0),(w,h), (0, 0, 0), 60)
#
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#                         cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cnts = contours.sort_contours(cnts)[0]
#
# # loop over the contours
# for (i, c) in enumerate(cnts):
#     # draw the bright spot on the image
#     (x, y, w, h) = cv2.boundingRect(c)
#     ((cX, cY), radius) = cv2.minEnclosingCircle(c)
#     cv2.circle(img, (int(cX), int(cY)), int(radius),
#                (0, 0, 255), 3)
#     cv2.putText(img, "#{}".format(i + 1), (x, y - 15),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large" components


# labels = measure.label(thresh, connectivity=2, background=0)
# mask = np.zeros(thresh.shape, dtype="uint8")
#
# # loop over the unique components
# for label in np.unique(labels):
#     # if this is the background label, ignore it
#     if label == 0:
#         continue
#
#     # otherwise, construct the label mask and count the
#     # number of pixels
#     labelMask = np.zeros(thresh.shape, dtype="uint8")
#     labelMask[labels == label] = 255
#     numPixels = cv2.countNonZero(labelMask)
#
#     # if the number of pixels in the component is sufficiently
#     # large, then add it to our mask of "large blobs"
#     if numPixels > 100:
#         mask = cv2.add(mask, labelMask)

# cv2.imshow("pointsDetection", img)

cv2.waitKey(0)& 0xFF== ord("q")
cv2.destroyAllWindows()