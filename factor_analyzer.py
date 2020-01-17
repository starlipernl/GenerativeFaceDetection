# Script for fitting and testing factor analyzer models for face detection.

from cv2 import imwrite
import numpy as np
from scipy import stats
from scipy.special import logsumexp
import utils


# function to fit the FA model
def fit_fa(imgs, k, max_iter):
    imgs = np.asarray(imgs)
    (n_imgs, n_pix) = np.shape(imgs)
    np.random.seed(7900)
    mu = np.mean(imgs, axis=0)
    # initialize covariances with full data variance
    sig = np.var(imgs, 0)
    # randomly initialize subspace
    phi = np.random.normal(size=(n_pix, k))
    imgs_center = imgs - mu
    e_hihi = np.zeros((k, k, n_imgs))
    # run EM up to the maximum number of iterations
    for i in range(0, max_iter):
        print(i)
        # Expectation step, calculate responsibilities using likelihoods
        phi_times_sig_inv = np.matmul(phi.T, np.diag(np.reciprocal(sig)))
        term1 = np.linalg.inv(np.matmul(phi_times_sig_inv, phi) + np.identity(k))
        e_hi = np.matmul(np.matmul(term1, phi_times_sig_inv), imgs_center.T)
        for img in range(0, n_imgs):
            e_hihi[:, :, i] = term1 + np.matmul(e_hi[:, img], e_hi[:, img].T)

        # M-Step
        phi_1 = np.matmul(imgs_center.T, e_hi.T)
        phi_2 = np.sum(e_hihi, 2)
        phi = np.matmul(phi_1, np.linalg.inv(phi_2))
        sig_sum = np.zeros((n_pix, 1))
        for img in range(0, n_imgs):
            img_c = imgs_center[img, :]
            sig_1 = np.multiply(img_c, img_c)
            sig_2 = np.multiply(np.matmul(phi, e_hi[:, img]), img_c)
            sig_sum = sig_sum + sig_1 - sig_2
        sig = np.diag(sig_sum / n_imgs)

    return phi, mu, sig


# function to predict new samples using trained FA model
def predict_fa(imgs, parms1, parms2):
    imgs = np.asarray(imgs)
    # calculate log likelihoods for each MoG component using both class models
    sig_tested = np.diag(parms1['sig']) + np.matmul(parms1['phi'], parms1['phi'].T)
    sig_notested = np.diag(parms2['sig']) + np.matmul(parms2['phi'], parms2['phi'].T)
    pr_xk_tested = stats.multivariate_normal.logpdf(imgs, parms1['mu'], sig_tested, allow_singular=True)
    pr_xk_nottested = stats.multivariate_normal.logpdf(imgs, parms2['mu'], sig_notested, allow_singular=True)

    pr_all = np.asarray([pr_xk_tested, pr_xk_nottested])
    sum_all = logsumexp(pr_all, axis=0)
    post = np.exp(pr_xk_tested - sum_all.T)
    return post


# resize because (60, 60) takes forever to run
size = (40, 40)
face_train_imgs, face_test_imgs, non_train_imgs, non_test_imgs = utils.load_data(size)
n_factors = 5
iters = 50

# fit models
phi_face, mu_face, sig_face = fit_fa(face_train_imgs, n_factors, iters)
phi_non, mu_non, sig_non = fit_fa(non_train_imgs, n_factors, iters)
params_face = {'phi': phi_face, 'mu': mu_face, 'sig': sig_face}
params_non = {'phi': phi_non, 'mu': mu_non, 'sig': sig_non}

# # test models on new data
post_face_face = predict_fa(face_test_imgs, params_face, params_non)
post_face_non = predict_fa(face_test_imgs, params_non, params_face)
post_non_non = predict_fa(non_test_imgs, params_non, params_face)
post_non_face = predict_fa(non_test_imgs, params_face, params_non)

# calculate misclassification rates
false_positives = np.sum(post_face_non > 0.5)
false_pos_rate = false_positives/100
false_negatives = np.sum(post_face_face < 0.5)
false_neg_rate = false_negatives/100
misclass_rate = (false_positives + false_negatives)/200

# plot ROC curve
utils.roc_plot(post_face_face, post_face_non)

# # stack component mean and covariance images for visualization
mu_all = np.hstack([np.reshape(mu_face * 255, size), np.reshape(mu_non * 255, size)])
sig_all = np.hstack([np.reshape(np.sqrt(sig_face) * 255, size), np.reshape(np.sqrt(sig_non) * 255, size)])

# visualize the different subspace factors
mu_factors = np.hstack([np.reshape(mu_face * 255 + 2 * p, size) for p in phi_face.T])
mu_factors_non = np.hstack([np.reshape(mu_non * 255 + 2 * p, size) for p in phi_non.T])
# write the mean and covariance faces to files for visualizations
imwrite('FA_Mean.png', mu_all)
imwrite('FA_Covariance.png', sig_all)
imwrite('FA_Means_Factors.png', mu_factors)
imwrite('FA_Means_Factors_Non_Face.png', mu_factors_non)
#
print("False Positive Rate: " + str(false_pos_rate))
print("False Negative Rate: " + str(false_neg_rate))
print("Misclassification Rate: " + str(misclass_rate))
