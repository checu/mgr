import cv2
import numpy as np
import sklearn.mixture
import scipy.spatial.distance
import argparse
import imutils
from imutils import contours
from scikit-image import measure

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


img = cv2.imread("testowy.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)


cv2.imshow("pointsDetection", img)

cv2.waitKey(0)& 0xFF== ord("q")
cv2.destroyAllWindows()