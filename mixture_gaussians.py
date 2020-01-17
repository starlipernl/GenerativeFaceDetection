import numpy as np
from scipy import stats
from scipy.special import logsumexp
from math import log
import utils
from cv2 import imwrite


# function to fit the MoG models
def fit_mog(imgs, k, tol, max_iter):
    imgs = np.asarray(imgs)
    lam = np.ones((k, 1)) * 1 / k
    (n_imgs, n_pix) = np.shape(imgs)
    np.random.seed(7900)
    # initialize component mu's using random images
    mu = imgs[np.random.randint(n_imgs, size=k), :]
    # initialize covariances with full data covariance
    cov_data = np.var(imgs, 0)
    # cov_data = np.diag(np.cov(np.transpose(imgs)))
    sig = np.tile(cov_data, (k, 1))
    previous_l = 1000000
    pr_xk = np.zeros((n_imgs, k))
    resp = np.zeros((n_imgs, k))
    L = np.zeros(max_iter)
    delta_l = np.zeros(max_iter)
    # run EM up to the maximum number of iterations or until converged
    for i in range(0, max_iter):
        print(i)
        # calculated log likelihoods for each component
        for comp in range(0, k):
            pr_xk[:, comp] = log(lam[comp]) + stats.multivariate_normal.logpdf(imgs, mu[comp, :], sig[comp, :],
                                                                               allow_singular=True)
        sum_pr_xk = logsumexp(pr_xk, axis=1)
        L[i] = np.sum(sum_pr_xk)  # total likelihood
        print(L[i])
        # find change in likelihoods
        delta_l[i] = abs(L[i] - previous_l)
        print(delta_l[i])
        previous_l = L[i]

        if delta_l[i] < tol:  # end optimization is change in likelihood converges
            break

        # Expectation step, calculate responsibilities using likelihoods
        for img in range(0, n_imgs):
            resp[img, :] = np.exp(pr_xk[img] - sum_pr_xk[img])
        sum_rik = np.sum(resp, 0)
        sum_r_all = np.sum(sum_rik)

        # Maximization Step, find new values for lambda, mu, and sigma using responsibilities
        for comp in range(0, k):
            lam[comp] = sum_rik[comp] / sum_r_all
            mu[comp, :] = np.matmul(resp[:, comp].transpose(), imgs) / sum_rik[comp]
            imgs_center = imgs - mu[comp, :]
            sig[comp, :] = np.matmul(np.transpose(resp[:, comp]), np.square(imgs_center)) / sum_rik[comp]

    return lam, mu, sig


# function to predict new samples using trained MoG models
def predict_mog(imgs, parms1, parms2):
    imgs = np.asarray(imgs)
    n_imgs_test = imgs.shape[0]
    k = parms1['lam'].shape[0]
    pr_xk_tested = np.zeros((n_imgs_test, k))
    pr_xk_nottested = np.zeros((n_imgs_test, k))
    # calculate log likelihoods for each MoG component using both class models
    for comp in range(0, k):
        pr_xk_tested[:, comp] = log(parms1['lam'][comp]) + stats.multivariate_normal.logpdf(imgs, parms1['mu'][comp, :],
                                                                                            parms1['sig'][comp, :],
                                                                                            allow_singular=True)
        pr_xk_nottested[:, comp] = log(parms2['lam'][comp]) + stats.multivariate_normal.logpdf(imgs, parms2['mu'][comp, :],
                                                                                               parms2['sig'][comp, :],
                                                                                               allow_singular=True)
    # calculate posteriors for each component
    sum_pr_xk_tested = np.asarray(logsumexp(pr_xk_tested, axis=1))
    sum_pr_xk_nottested = np.asarray(logsumexp(pr_xk_nottested, axis=1))
    pr_all = np.asarray([sum_pr_xk_tested.T, sum_pr_xk_nottested.T])
    sum_all = logsumexp(pr_all, axis=0)
    post = np.exp(pr_xk_tested.T - sum_all)
    return post


# resize because (60, 60) takes forever to run
size = (40, 40)
face_train_imgs, face_test_imgs, non_train_imgs, non_test_imgs = utils.load_data(size)
K = 5
tolerance = 0.001
iters = 100
# fit models
lam_face, mu_face, sig_face = fit_mog(face_train_imgs, K, tolerance, iters)
lam_non, mu_non, sig_non = fit_mog(non_train_imgs, K, tolerance, iters)
params_face = {'lam': lam_face, 'mu': mu_face, 'sig': sig_face}
params_non = {'lam': lam_non, 'mu': mu_non, 'sig': sig_non}
# test models on new data
post_face_face = predict_mog(face_test_imgs, params_face, params_non)
post_face_non = predict_mog(face_test_imgs, params_non, params_face)
post_non_non = predict_mog(non_test_imgs, params_non, params_face)
post_non_face = predict_mog(non_test_imgs, params_face, params_non)
# sum posteriors of all mixture components for each image
post_ff_sum = np.sum(post_face_face, 0)
post_fn_sum = np.sum(post_face_non, 0)
# calculate misclassification rates
false_positives = np.sum(post_fn_sum > 0.5)
false_pos_rate = false_positives/100
false_negatives = np.sum(post_ff_sum < 0.5)
false_neg_rate = false_negatives/100
misclass_rate = (false_positives + false_negatives)/200
# plot ROC curve
utils.roc_plot(post_ff_sum, post_fn_sum)
# stack component mean and covariance images for visualization
mu_face_stack = np.hstack([np.reshape(m * 255, size) for m in mu_face])
mu_non_stack = np.hstack([np.reshape(m * 255, size) for m in mu_non])
mu_all = np.vstack([mu_face_stack, mu_non_stack])
sig_face_stack = np.hstack([np.reshape(np.sqrt(s) * 255, size) for s in sig_face])
sig_non_stack = np.hstack([np.reshape(np.sqrt(s) * 255, size) for s in sig_non])
sig_all = np.vstack([sig_face_stack, sig_non_stack])

# write the mean and covariance faces to files for visualizations
imwrite('MoG_Mean.png', mu_all)
imwrite('MoG_Covariance.png', sig_all)

print("False Positive Rate: " + str(false_pos_rate))
print("False Negative Rate: " + str(false_neg_rate))
print("Misclassification Rate: " + str(misclass_rate))
