# Script for fitting and testing t-distribution models for face detection.

from cv2 import imwrite
import numpy as np
from scipy.special import digamma, gammaln
from scipy.optimize import fminbound
from scipy.special import logsumexp
import utils
from math import log


# function to fit the FA model
def fit_fa(imgs, max_iter):
    imgs = np.asarray(imgs)
    (n_imgs, d) = np.shape(imgs)
    np.random.seed(7900)
    mu = np.mean(imgs, axis=0)
    # initialize covariances with full data variance
    sig = np.var(imgs, 0)
    # randomly initialize subspace
    nu = 1000
    delta = np.zeros((n_imgs, 1))
    # run EM up to the maximum number of iterations
    for i in range(0, max_iter):
        print(i)
        # Expectation step, calculate responsibilities using likelihoods
        imgs_center = imgs - mu
        imgs_center_x_sig = imgs_center @ np.diag(np.reciprocal(sig))
        for img in range(0, n_imgs):
            delta[img] = imgs_center_x_sig[img, :] @ imgs_center[img, :].T

        e_hi = np.divide((nu + d), (nu + delta))
        e_log_hi = digamma((nu + d) / 2) - np.log((nu + delta) / 2)

        # M-Step
        e_hi_sum = np.sum(e_hi)
        mu = np.sum(np.multiply(e_hi, imgs), 0) / e_hi_sum
        imgs_center = imgs - mu
        e_hi_x_imgs_c = np.multiply(e_hi, imgs_center) # 1000x100
        sig = np.diag(e_hi_x_imgs_c.T @ imgs_center / e_hi_sum)
        [_, logdet] = np.linalg.slogdet(np.diag(sig))
        for img in range(0, n_imgs):
            delta[img] = imgs_center_x_sig[img, :] @ imgs_center[img, :].T
        nu = fminbound(t_cost, 0, 1000, (e_hi, e_log_hi, delta, logdet))
    return nu, mu, sig


# function to predict new samples using trained FA model
def predict_fa(imgs, parms1, parms2):
    imgs = np.asarray(imgs)
    # calculate log likelihoods for each MoG component using both class models
    (n_imgs, d) = np.shape(imgs)
    dictget = lambda d, *k: [d[i] for i in k]
    nu1, mu1, sig1 = dictget(parms1, 'nu', 'mu', 'sig')
    nu2, mu2, sig2 = dictget(parms2, 'nu', 'mu', 'sig')

    # likelihood of true class
    imgs_center = imgs - mu1
    imgs_center_x_sig = imgs_center @ np.diag(np.reciprocal(sig1))
    delta = np.zeros((n_imgs, 1))
    for img in range(0, n_imgs):
        delta[img] = imgs_center_x_sig[img, :] @ imgs_center[img, :].T
    [_, logdet] = np.linalg.slogdet(np.diag(sig1))
    pr_xk_true = gammaln((nu1 + d) / 2) - (d / 2) * np.log(nu1 * np.pi) - logdet / 2 - gammaln(nu1 / 2) - 0.5 * (nu1 + d) * np.log(1 + delta / nu1)
    # likelihood of false class
    imgs_center = imgs - mu2
    imgs_center_x_sig = imgs_center @ np.diag(np.reciprocal(sig2))
    delta = np.zeros((n_imgs, 1))
    for img in range(0, n_imgs):
        delta[img] = imgs_center_x_sig[img, :] @ imgs_center[img, :].T
    [_, logdet] = np.linalg.slogdet(np.diag(sig2))
    pr_xk_false = gammaln((nu2 + d) / 2) - (d / 2) * np.log(nu2 * np.pi) - logdet / 2 - gammaln(nu2 / 2) - 0.5 * (nu2 + d) * np.log(1 + delta / nu2)
    pr_all = np.asarray([pr_xk_true, pr_xk_false])
    sum_all = logsumexp(pr_all, axis=0)
    post = np.exp(pr_xk_true - sum_all)
    return post


def t_cost(nu, e_hi, e_log_hi, delta, logdet):
    nu_half = nu / 2
    D = e_hi.shape[0]
    cost = D * (nu_half * log(nu_half) * gammaln(nu_half))
    cost -= (nu_half - 1) * np.sum(e_log_hi)
    cost += nu_half * np.sum(e_hi)
    term1 = (D * np.sum(e_log_hi) - D * log(2*np.pi) - logdet - delta * e_hi)/2
    cost += np.sum(term1)
    return cost


# resize because (60, 60) takes forever to run
size = (10, 10)
face_train_imgs, face_test_imgs, non_train_imgs, non_test_imgs = utils.load_data(size)
iters = 50

# fit models
nu_face, mu_face, sig_face = fit_fa(face_train_imgs, iters)
nu_non, mu_non, sig_non = fit_fa(non_train_imgs, iters)
params_face = {'nu': nu_face, 'mu': mu_face, 'sig': sig_face}
params_non = {'nu': nu_non, 'mu': mu_non, 'sig': sig_non}

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

# write the mean and covariance faces to files for visualizations
# imwrite('t_Meanb.png', mu_all)
# imwrite('t_Covarianceb.png', sig_all)

#
print("False Positive Rate: " + str(false_pos_rate))
print("False Negative Rate: " + str(false_neg_rate))
print("Misclassification Rate: " + str(misclass_rate))

