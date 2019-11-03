import cv2
import numpy as np

img = cv2.imread("images/Hand_0000282.jpg")
img = cv2.resize(img, (360, 480))


def getHandFormImage(img):
    # converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # skin color range for YCrCb color space
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # inverst YCrCb mask
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)

    final = cv2.bitwise_not(img, img, YCrCb_result)
    return final


def calculatingMeanForYCrCbModelChanels(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

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

    return yMean, yMax, yMin, yStd, crMean, crMax, crMin, crStd, cbMean, cbMax, cbMin, cbStd
