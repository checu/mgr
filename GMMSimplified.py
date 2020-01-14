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


    if (mahalanobisDistanceClass0Min + mahalanobisDistanceClass0Max < mahalanobisDistanceClass1Min + mahalanobisDistanceClass1Max): #& ((labels == 0).sum() > (labels == 1).sum()):
        print(0)
        return 0
    else:
        print(1)
        return 1


def GMM(img, image_path):

# input image preporcessing - resize

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

    cv2.imwrite(image_path, img_RGB_after)



#BASIC OPPERATIONS

image_input_directory = ""
image_output_directory = ""

image_list_full = os.listdir(image_input_directory)

image_list = random.choices(image_list_full, k=50)


#process data

for image_name in image_list_full:

    image_path = image_output_directory + image_name
    image = cv2.imread(image_input_directory + image_name)
    GMM(image, image_path)


cv2.waitKey(0)& 0xFF== ord("q")
cv2.destroyAllWindows()
