import cv2
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from math import exp
import utils


# function to find gaussian parameters (mean/covariance matrix)
def gauss_parms(img):
    mean = np.mean(img, axis=0)
    covar = np.cov(np.transpose(img))
    return mean, covar


# function to calculate posterior probability using the log sum exp trick
def post_prob(pr_x0, pr_x1):
    post_0x = np.zeros(len(pr_x0))
    post_1x = np.zeros(len(pr_x1))
    for pix in range(len(pr_x0)):
        post_0x[pix] = exp(pr_x0[pix] - logsumexp([pr_x0[pix], pr_x1[pix]]))
        post_1x[pix] = exp(pr_x1[pix] - logsumexp([pr_x0[pix], pr_x1[pix]]))
    return post_0x, post_1x


# main
size = (60, 60)
face_train_imgs, face_test_imgs, non_train_imgs, non_test_imgs = utils.load_data(size)
# find mean and covariance of training images to build gauss model
m_face, cov_face = gauss_parms(face_train_imgs)
m_non, cov_non = gauss_parms(non_train_imgs)

cov_nd = np.diag(np.diag(cov_non))
cov_fd = np.diag(np.diag(cov_face))

# find the logpdf in order to exploit the log-sum-exp trick
pr_f0 = stats.multivariate_normal.logpdf(face_test_imgs, m_non, cov_nd)
pr_f1 = stats.multivariate_normal.logpdf(face_test_imgs, m_face, cov_fd)
pr_n0 = stats.multivariate_normal.logpdf(non_test_imgs, m_non, cov_nd)
pr_n1 = stats.multivariate_normal.logpdf(non_test_imgs, m_face, cov_fd)

# calculate posteriors
post_0f, post_1f = post_prob(pr_f0, pr_f1)
post_0n, post_1n = post_prob(pr_n0, pr_n1)

# calculate mislcassification rates
false_positives = np.sum(post_1n > 0.5)
false_pos_rate = false_positives/100
false_negatives = np.sum(post_1f < 0.5)
false_neg_rate = false_negatives/100
misclass_rate = (false_positives + false_negatives)/200

# plot ROC curve
utils.roc_plot(post_1f, post_1n)

mu_all = np.hstack([np.reshape(m_face * 255, size), np.reshape(m_non * 255, size)])
cov_all = np.hstack([np.reshape(np.sqrt(np.diag(cov_face)), size), np.reshape(np.sqrt(np.diag(cov_non)), size)])
# scale 0 to 255
cov_all *= 255./np.amax(cov_all)
# write the mean and covariance faces to files for visualizations
cv2.imwrite('Gaussian_Mean.png', mu_all)
cv2.imwrite('Gaussian_Cov.png', cov_all)

print("False Positive Rate: " + str(false_pos_rate))
print("False Negative Rate: " + str(false_neg_rate))
print("Misclassification Rate: " + str(misclass_rate))
