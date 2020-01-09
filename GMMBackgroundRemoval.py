import cv2
import numpy as np
import sklearn.mixture
import scipy.spatial.distance
import os
import random



def getBackgroundLabel (means, precisions, labels):

    precisions_diag0 = np.diag(precisions[0])
    precisions_diag1 = np.diag(precisions[1])

    #YcrCb
    yCrCbHandColorRangeAvg = np.array([127.5, 157.5, 110])
    yCrCbHandColorRangeMin = np.array([0, 135, 85])
    yCrCbHandColorRangeMax = np.array([255, 180, 135])

    #HSV
    # yCrCbHandColorRangeAvg = np.array([8.5, 92.5, 127.5])
    # yCrCbHandColorRangeMin = np.array([0, 15, 0])
    # yCrCbHandColorRangeMax = np.array([17, 170, 255])

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

    if (mahalanobisDistanceClass0Min + mahalanobisDistanceClass0Max < mahalanobisDistanceClass1Min + mahalanobisDistanceClass1Max) & ((labels == 0).sum() > (labels == 1).sum()):
        print(0)
        return 0
    else:
        print(1)
        return 1

def mommentum(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=2)

    moments = cv2.moments(edged)
    hu = cv2.HuMoments(moments)
    centres = []
    centres.append((int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])))

    cv2.circle(img, centres[-1], 3, (0, 0, 255), -1)

    momentPoint = centres[-1]

    # cv2.imshow("Momentum img", img)

    return momentPoint

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

def biggestInscribedCircle(img):

    # Image preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=2)

    # Finding the circle
    VPResults = FindBiggestContour(edged)
    dist = 0
    maxdist = 0

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dist = cv2.pointPolygonTest(VPResults, (i, j), True)
            if dist > maxdist:
                maxdist = dist
                center = (i, j)

    cv2.circle(img, center, int(maxdist), (0, 255, 255), 2)
    cv2.circle(img, center, 3, (0, 255, 0), -1)

    # cv2.rectangle(img, tuple(c - int(maxdist) for c in center), tuple(c + int(maxdist) for c in center), (255, 255, 255), 2)

    # cv2.imshow("Inscribed Circle img", img)

    return center



def GMM(img, image_path):

# input image preporcessing

    img = cv2.resize(img, (360, 480))

    # cv2.imshow("input_image", img)

# YCrCb
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# HSV
#   img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    y, cr, cb = img_YCrCb[:, :, 0], img_YCrCb[:, :, 1], img_YCrCb[:, :, 2]

#GMM

    GMM = sklearn.mixture.GaussianMixture (n_components=2, covariance_type='diag', tol=0.001, reg_covar=1e-06,
                                 max_iter=100, n_init=1, init_params='random', weights_init=None,
                                 means_init=None, precisions_init=None, random_state=None, warm_start=False,
                                 verbose=0, verbose_interval=10)

#reshape image to 2d arrray

    h, w, ycrcb = img_YCrCb.shape
    d2_train_dataset = img_YCrCb.reshape((h*w,ycrcb))

#  -----------GMM-----------
    GMM_results = GMM.fit(d2_train_dataset)

#GMM parameters

    GMM_means = GMM.means_
    GMM_covariance = GMM.covariances_
    GMM_precision = GMM.precisions_

#Background label detection

    lables= GMM.fit_predict(d2_train_dataset)

    print("Labels :", lables)

    replace2d_dataset= d2_train_dataset.copy()

    print(os.path.basename(image_path))

    label = getBackgroundLabel(GMM_means, GMM_precision, lables)

    replace2d_dataset[lables.astype(bool) == label] = [16,128,128]
    # replace2d_dataset[lables.astype(bool) == label] = [0,0,0]

    reshaped_YCrCb = replace2d_dataset.reshape(h,w,ycrcb)


# YCrCb image into RGB & Ensure all background is black RGB = [0,0,0]

    img_RGB_after = cv2.cvtColor(reshaped_YCrCb, cv2.COLOR_YCrCb2BGR)

    img_RGB_after[img_RGB_after == [16,16,16]] = 0

# HSV image into RGB & Ensure all background is black RGB = [0,0,0]

    # img_RGB_after = cv2.cvtColor(reshaped_YCrCb, cv2.COLOR_HSV2BGR)
    #
    # img_RGB_after[img_RGB_after == [0,0,0]] = 0

    # cv2.imshow("before smoothing_image", img_RGB_after)


# Holes smoothing

    gray = cv2.cvtColor(img_RGB_after, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)[1]

    inpaint = cv2.inpaint(img_RGB_after, thresh, 3, cv2.INPAINT_TELEA)

    kernel = np.ones((6,6),np.uint8)

    closing = cv2.bitwise_not(thresh)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    closing = cv2.dilate(closing, None, iterations=2)
    closing = cv2.erode(closing, None, iterations=2)

    clop = np.where(closing[...,None] == 0 , [255,255,255],[0,0,0])

    output_image = np.where(clop == np.array([0,0,0]),inpaint,img_RGB_after)

    cv2.imwrite(image_path, output_image)


    temp_image = output_image.copy()

    # cv2.imshow("output img", temp_image)

    momentCoordinates = mommentum(temp_image)

    inscribedCircleCoordinates = biggestInscribedCircle(output_image)

    return momentCoordinates, inscribedCircleCoordinates

f= open("guru99.txt","w+")


#main

image_input_directory = "/Users/chekumis/Desktop/Palmar/"
image_output_directory = "/Users/chekumis/Desktop/PalmarBBGtest/"

image_list_full = os.listdir(image_input_directory)

image_list = random.choices(image_list_full, k=50)


#dataFile

f = open("/Users/chekumis/Desktop/HandExpansionCoordinates.txt",'x')

for image_name in image_list:

    image_path = image_output_directory + image_name
    image = cv2.imread(image_input_directory + image_name)

    momentCoord, circleCoord = GMM(image, image_path)

    f.write(image_name + ",%s,%s \n" %(momentCoord, circleCoord))

f.close()


cv2.waitKey(0)& 0xFF== ord("q")
cv2.destroyAllWindows()