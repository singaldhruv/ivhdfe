import numpy as np
import scipy as sp
import pandas as pd

from .fixed_effects import demean_FE
from .test_statistics import calc_f_stats

# Use QR decomposition to first get Xq @ Xr = X
# If no y is provided, just returns (X'X)^-1
# If y is provided, solve OLS: y = X @ beta + e
def ols_solve(X, y=None):
    if X.shape[1] == 0:
        raise ValueError('Must have at least covariate for valid OLS')
    Xq, Xr = np.linalg.qr(X)
    gram_inv = np.linalg.solve(Xr.T @ Xr, np.eye(X.shape[1]))
    if y is None:
        return gram_inv
    beta = sp.linalg.solve_triangular(Xr, Xq.T @ y)
    fitted = X @ beta 
    resid = y - fitted
    return beta, gram_inv, fitted, resid

# First stage: X = [W, Z] @ alpha + u
# Second stage: y = [W, Xhat] @ beta + e
def tsls_solve(X, W, Z, y, iv_params, fe_params=None):
    N = Z.shape[0]
    k = Z.shape[1]
    if (W.shape[1] == 0) and (fe_params is None):
        raise ValueError('Must have at least one control or fixed effect for valid TSLS')
    vcov_params = iv_params['vcov_params']
    resid_dof = iv_params['resid_dof']
    fs_resid_dof = resid_dof + X.shape[1] - k
    
    WZ = np.hstack([W, Z])
    Xhat = np.zeros_like(X)
    resid_first = np.zeros_like(X)
    f_stats = np.zeros((X.shape[1],1))
    pi = np.zeros((k, X.shape[1]))
    
    for x_idx in range(X.shape[1]):
        curr_X = X[:,x_idx][:,np.newaxis]
        ret_vals = fe_solve(WZ, curr_X, fe_params)
        curr_beta, curr_WZ_gram_inv, curr_Xhat, curr_resid_first, curr_WZ, _ = ret_vals 
        pi[:,x_idx] = np.squeeze(curr_beta[-k:,:])
        Xhat[:,x_idx] = np.squeeze(curr_Xhat)
        resid_first[:,x_idx] = np.squeeze(curr_resid_first)
        
    print(calc_f_stats(pi, curr_WZ_gram_inv, curr_WZ, 
                        resid_first, fs_resid_dof,
                        vcov_params=vcov_params))
    
    f_stats = calc_f_stats(pi, curr_WZ_gram_inv, curr_WZ, 
                        resid_first, fs_resid_dof,
                        vcov_params={'vcov_type': 'iid'})
    f_stats = np.reshape(np.diag(f_stats), (1,-1))
    
    
    WXhat = np.hstack([W, Xhat])
    if fe_params is None: 
        ret_vals = ols_solve(WXhat, y)
        ret_vals += (WXhat, None)
    else:
        ret_vals = fe_solve(WXhat, y, fe_params)
    beta, gram_inv, fitted, resid_second, reg_covariates, y_demeaned = ret_vals
    resid = resid_second - resid_first @ beta[-X.shape[1]:,:] 
    return beta, gram_inv, fitted, resid, reg_covariates, y_demeaned, f_stats

# Assumes Gaussian/OLS: Solve OLS using demeaned X and y
def fe_solve(X, y=None, fe_params=None):
    if fe_params is None:
        if y is None:
            return ols_solve(X)
        else:
            return ols_solve(X, y) + (X, y)
    else:
        if y is None:
            y_demeaned = None
        else:
            y_demeaned = demean_FE(y, fe_params['FE_matrix'], fe_params['groups_FE'], fe_params['levels_FE'], fe_params['obs_levels_FE'], fe_params['init_fitted_FE'])
            if X.shape[1] == 0:
                return None, None, y - y_demeaned, y_demeaned, None, y_demeaned
            
        X_demeaned = np.zeros_like(X)
        for k in range(X.shape[1]):
            curr_X = X[:,k][:,np.newaxis]
            curr_X_demeaned = demean_FE(curr_X, fe_params['FE_matrix'], fe_params['groups_FE'], fe_params['levels_FE'], fe_params['obs_levels_FE'], fe_params['init_fitted_FE'])
            X_demeaned[:,k] = np.squeeze(curr_X_demeaned)
        
        beta, gram_inv, fitted, resid = ols_solve(X_demeaned, y_demeaned)
        return beta, gram_inv, fitted, resid, X_demeaned, y_demeaned