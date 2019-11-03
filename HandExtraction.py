import cv2
import numpy as np
import imutils

bg = None



#SkinDetektor

hLowTreshold = 0
hHighTreshold = 0
sLowTreshold = 0
sHighTreshold = 0
vLowTreshold = 0
vHighTreshold = 0

calibrated = False



image = cv2.imread('images/Hand_0000106.jpg')

imager = cv2.resize(image, (360,480))



#tresholding

gray = cv2.cvtColor(imager,cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray,(5,5),0)

ret,tresh1 = cv2.threshold(blur,10,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#cv2.imshow("Picture", imager)

cv2.imshow("Tresholded", tresh1)

cv2.waitKey(0) # waits until a key is pressed

cv2.destroyAllWindows()