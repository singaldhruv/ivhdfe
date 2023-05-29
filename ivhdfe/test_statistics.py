import numpy as np

from .vcov import calc_vcov

def calc_f_stats(pi, WZ_gram_inv, WZ, resid_first, fs_resid_dof, vcov_params, f_stat_type=None):
    k = pi.shape[0]
    if f_stat_type is None:
        # Robust F-statistic using the same assumptions as standard errors
        pi_vcov_fun = lambda curr_resid_first: calc_vcov(WZ_gram_inv, WZ, curr_resid_first, fs_resid_dof, vcov_params)[-k:,-k:]
    elif f_stat_type == 'CD':
        # Cragg-Donald F-statistic (assumes homoskedasticity)
        pi_vcov_fun = lambda curr_resid_first: calc_vcov(WZ_gram_inv, WZ, curr_resid_first, fs_resid_dof, {'vcov_type': 'iid'})[-k:,-k:]
    elif f_stat_type == 'KP':
        # Kleibergen-Paap F-statistic (assumes heteroskedasticity)
        pi_vcov_fun = lambda curr_resid_first: calc_vcov(WZ_gram_inv, WZ, curr_resid_first, fs_resid_dof, {'vcov_type': 'robust'})[-k:,-k:]
    else:
        raise ValueError('f_stat_type must be None, CD, or KP')
    
    def curr_f_stat(curr_pi, curr_resid_first):
        pi_vcov = pi_vcov_fun(curr_resid_first)
        return 1/k * curr_pi[:,0].T @ np.linalg.inv(pi_vcov) @ curr_pi[:,1]

    # TODO: This logic doesn't work for p > 1
    # It returns the correct F-statistics on the diagonals
    # But the min eigenvalue doesn't match Stata/R
    p = resid_first.shape[1]
    f_stats = np.zeros((p,p))
    for ii in range(p):
        for jj in range(ii, p):
            curr_resid_first = resid_first[:,[ii,jj]]
            curr_pi = pi[:,[ii,jj]]
            f_stats[ii,jj] = np.squeeze(curr_f_stat(curr_pi, curr_resid_first))
            f_stats[jj,ii] = f_stats[ii,jj]
    return f_stats