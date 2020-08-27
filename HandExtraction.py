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



image = cv2.imread('images/Hand_0000254.jpg')

imager = cv2.resize(image, (360,480))

cv2.imshow("Original Image", imager)



#tresholding

gray = cv2.cvtColor(imager,cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray", gray)

blur = cv2.GaussianBlur(gray,(5,5),0)
blur2 = cv2.GaussianBlur(gray,(3,3),0)
blur3 = cv2.GaussianBlur(gray,(7,7),0)

cv2.imshow("Blur", blur3)

blur4 = cv2.GaussianBlur(gray,(13,13),0)

ret,tresh1 = cv2.threshold(blur,10,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

ret1,tresh2 = cv2.threshold(blur2,10,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

ret3,tresh3 = cv2.threshold(blur3,10,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

ret4,tresh4 = cv2.threshold(blur4,10,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#cv2.imshow("Picture", imager)

cv2.imshow("Tresholded 5x5", tresh1)
cv2.imshow("Tresholded 3x3", tresh2)
cv2.imshow("Tresholded 7x7", tresh3)
cv2.imshow("Tresholded 13x13", tresh4)


cv2.waitKey(6000000) # waits until a key is pressed

cv2.destroyAllWindows()