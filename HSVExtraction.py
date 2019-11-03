import cv2
import imutils
import numpy as np


# Grab a picture

image = cv2.imread('images/Hand_0000106.jpg')

img = cv2.resize(image, (400, 700))

clone = img.copy()


# Detect Skin Tone

def DetectSkinTone(img):

    hLowThreshold = 0
    hHighThreshold = 0
    sLowThreshold = 0
    sHighThreshold = 0
    vLowThreshold = 0
    vHighThreshold = 0

#draw Sample rectangle

    (frameHeight, frameWidth) = img.shape[:2]

    rectangleSize = 20

    rectangleColor = (255,0,255)

    skinColorSample = cv2.rectangle(img, (230, 175), (300, 270), (0,0,255), 2) #tutaj trzeba ogarnać, jak to zrobic bardziej flexible

    cv2.imshow("FillPolly", img)


#calibration

    hsvInput = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

    sample = np.array([hsvInput,skinColorSample], dtype='uint8')

#calculating threshold

    offsetLowThreshold = 80
    offsetHighThreshold = 30

    hsvMeansSample = cv2.mean(sample)
    #
    # hLowThreshold = min(hsvMeansSample[0]) - offsetLowThreshold
    # hHighThreshold = max(hsvMeansSample[0]) + offsetHighThreshold
    #
    # sLowThreshold = min(hsvMeansSample[1]) - offsetLowThreshold
    # sHighThreshold = max(hsvMeansSample[1]) + offsetHighThreshold
    #
    # vLowThreshold = min(hsvMeansSample[2]) - offsetLowThreshold
    # vHighThreshold = max(hsvMeansSample[2]) + offsetHighThreshold

#Get skin mask



def BackgroundRemover():
    pass

DetectSkinTone(clone)

cv2.waitKey(0)

cv2.destroyAllWindows()