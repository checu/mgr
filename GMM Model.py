import cv2
import numpy as np
import sklearn.mixture
import scipy.spatial.distance
import matplotlib.pyplot as plt
import imutils


img = cv2.imread("images/Hand_0000345.jpg")
img = cv2.resize(img, (360, 480))

# cv2.imshow("Final", img)

img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# extract chanels from image
y, cr, cb = img_YCrCb[:, :, 0], img_YCrCb[:, :, 1], img_YCrCb[:, :, 2]


def GetBackgroundLabel (means, precisions, labels):

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

    if (mahalanobisDistanceClass0Min + mahalanobisDistanceClass0Max < mahalanobisDistanceClass1Min + mahalanobisDistanceClass1Max) & ((lables == 0).sum() > (lables == 1).sum()):
        print(0)
        return 0
    else:
        print(1)
        return 1

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

label = GetBackgroundLabel(GMM_means,GMM_precision, lables)

replace2d_dataset[lables.astype(bool) == label] = [16,128,128]


reshaped_YCrCb = replace2d_dataset.reshape(h,w,ycrcb)

img_YCrCb_after = cv2.cvtColor(reshaped_YCrCb, cv2.COLOR_YCrCb2BGR)


cv2.imshow("after GMM", img_YCrCb_after)

# edge smoothing

img_flatten = img_YCrCb_after.copy()

img_blur = cv2.GaussianBlur(img_flatten, (5,5),0)
mask = np.zeros(img_flatten.shape, np.uint8)

gray = cv2.cvtColor(img_flatten, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)[1]

cv2.imshow("tresholded image", thresh)

kernel = np.ones((3,3),np.uint8)

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(mask, contours, -1, (255,255,255),5)

cv2.imshow("contours mask", mask)

closing = cv2.inpaint(img_flatten,thresh,10,cv2.INPAINT_NS)

cv2.imshow("closed image", closing)

output = np.where(mask==np.array([255, 255, 255]), img_blur, img_flatten)

cv2.imshow("after flatten", output)

#cut the fingers

# gray = cv2.cvtColor(img_YCrCb_after, cv2.COLOR_BGR2GRAY)
#
# gray = cv2.GaussianBlur(gray, (7, 7), 0)
#
# edged = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
# edged = cv2.dilate(edged, None, iterations=2)
# edged = cv2.erode(edged, None, iterations=2)
#
#
#
# cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0]
# c = max(cnts, key=cv2.contourArea)
# hull = cv2.convexHull(c,returnPoints = False)
# defects = cv2.convexityDefects(c,hull)
#
# for i in range(defects.shape[0]):
#     s,e,f,d = defects[i,0]
#     start = tuple(c[s][0])
#     end = tuple(c[e][0])
#     far = tuple(c[f][0])
#     cv2.line(img,start,end,[0,255,0],2)precisions_diag1 = precisions[0].diag(precisions)
#     cv2.circle(img,far,5,[0,0,255],-1)
#
# cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
# cv2.imshow("Image", img)


cv2.waitKey(0)& 0xFF== ord("q")
cv2.destroyAllWindows()



#calculating mehanalobis distance for the picture points

# labels = GMM.predict(d2_train_dataset)
#
# plt.scatter(d2_train_dataset[:, 1], d2_train_dataset[:, 2], c = labels, s= 30, cmap='viridis');
#
# plt.show()
