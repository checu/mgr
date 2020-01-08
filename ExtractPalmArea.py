import cv2

import numpy as np
import math


circleCenter = (148,70)

momentCoord = (155,169)

startPoint = (138,192)

endPoint = (211,151)


img = cv2.imread("testowy233.jpg")

# cv2.line(img, startPoint, endPoint, (0, 250, 0), 2)

# cv2.line(img, circleCenter, momentCoord, (0, 250, 0), 2)

distHorizontal = math.sqrt((momentCoord[0]-circleCenter[0])**2 + (momentCoord[1]-circleCenter[1])**2)

distVertical = math.sqrt((startPoint[0]-endPoint[0])**2 + (startPoint[1]-endPoint[1])**2)


a = ((circleCenter[1]-momentCoord[1])/(circleCenter[0]-momentCoord[0]))


b1 = startPoint[1] - a*startPoint[0]

b2 = endPoint[1] - a*endPoint[0]

# calculate alpha
angleRadians = math.atan(a)

angleDegrees = math.degrees(angleRadians)

cx = int(distVertical * math.cos(angleRadians))

cy = int(distVertical * math.sin(angleRadians))

dx = int(distVertical * math.cos(angleRadians))

dy = int(distVertical * math.sin(angleRadians))

if a > 0 :
    C = (startPoint[0] - cx, startPoint[1] - cy)

    D = (endPoint[0] - dx, endPoint[1] - dy)

else:

    C = (startPoint[0] + cx, startPoint[1] - cy)

    D = (endPoint[0] + dx, endPoint[1] - dy)


points = np.array([startPoint,C,D,endPoint])

cv2.polylines(img,[points],True,(0,255,255))
# cv2.fillPoly(img,[points],(255,255,255))

# cv2.line(img, startPoint, C, (0, 250, 0), 2)
# cv2.line(img, endPoint, D, (0, 250, 0), 2)
# cv2.line(img, C, D, (0, 250, 0), 2)
# wspolczynnik korelacji dla szukanych wspolrzednych

mask = np.zeros(img.shape, dtype=np.uint8)

roi_corners = np.array(points, dtype=np.int32)

cv2.fillPoly(mask, [points], (255,255,255))

masked_image = cv2.bitwise_and(img, mask)


# cv2.circle(img,C, 2, (255, 0, 255), -1)
# cv2.circle(img,D, 2, (255,255, 255), -1)


cv2.imshow("Cross", masked_image)





cv2.waitKey(0)& 0xFF== ord("q")
cv2.destroyAllWindows()