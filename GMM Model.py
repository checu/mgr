import cv2
import numpy as np
import sklearn.mixture
import scipy.spatial.distance
import matplotlib.pyplot as plt


img = cv2.imread("images/Hand_0000106.jpg")
img = cv2.resize(img, (360, 480))

# cv2.imshow("Final", img)

img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# extract chanels from image
y, cr, cb = img_YCrCb[:, :, 0], img_YCrCb[:, :, 1], img_YCrCb[:, :, 2]


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

lables= GMM.fit_predict(d2_train_dataset)

replace2d_dataset= d2_train_dataset.copy()


replace2d_dataset[lables.astype(bool)] = [16,128,128]


reshaped_YCrCb = replace2d_dataset.reshape(h,w,ycrcb)

img_YCrCb_after = cv2.cvtColor(reshaped_YCrCb, cv2.COLOR_YCrCb2BGR)


cv2.imshow("after GMM", img_YCrCb_after)

cv2.waitKey(0)
cv2.destroyAllWindows()



#calculating mehanalobis distance for the picture points

# labels = GMM.predict(d2_train_dataset)
#
# plt.scatter(d2_train_dataset[:, 1], d2_train_dataset[:, 2], c = labels, s= 30, cmap='viridis');
#
# plt.show()
