import numpy as np
import scipy as sp
import pandas as pd

from src.fixed_effects import demean_FE
from src.vcov import calc_vcov_iid, calc_vcov_hc1

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
    k_controls = W.shape[1]
    if (k_controls == 0) and (fe_params is None):
        raise ValueError('Must have at least one control or fixed effect for valid TSLS')
    vcov_fun = iv_params['vcov_fun']
    resid_dof = iv_params['resid_dof']
    fs_resid_dof = resid_dof + X.shape[1] - k
    
    WZ = np.hstack([W, Z])
    Xhat = np.zeros_like(X)
    resid_first = np.zeros_like(X)
    resid_first_no_inst = np.zeros_like(X)
    f_stats = np.zeros((X.shape[1],1))
    beta_first = np.zeros((WZ.shape[1], X.shape[1]))
    pi = np.zeros((k, X.shape[1]))
    
    for x_idx in range(X.shape[1]):
        curr_X = X[:,x_idx][:,np.newaxis]
        ret_vals = fe_solve(WZ, curr_X, fe_params)
        ret_vals_no_inst = fe_solve(W, curr_X, fe_params)
        curr_beta, curr_WZ_gram_inv, curr_Xhat, curr_resid_first, curr_WZ, _ = ret_vals 
        beta_first[:,x_idx] = np.squeeze(curr_beta)
        pi[:,x_idx] = np.squeeze(curr_beta[k_controls:,:])
        Xhat[:,x_idx] = np.squeeze(curr_Xhat)
        resid_first[:,x_idx] = np.squeeze(curr_resid_first)
        
        _, _, _, curr_resid_first_no_inst, _, _ = ret_vals_no_inst
        resid_first_no_inst[:,x_idx] = np.squeeze(curr_resid_first_no_inst)
        curr_ssr_first = np.sum(curr_resid_first**2)
        curr_ssr_first_no_inst = np.sum(curr_resid_first_no_inst**2)
        curr_f_stat = (curr_ssr_first_no_inst - curr_ssr_first)  / curr_ssr_first
        f_stats[x_idx] = curr_f_stat*(fs_resid_dof/k)
        
    # This is close to Stata, and less than R
    # Cragg-Donald F-statistic
    Sigma_uu = (X.T @ resid_first)/(fs_resid_dof)
    Sigma_uu_inv = np.linalg.inv(Sigma_uu)
    Sigma_uu_sqrt_inv = sp.linalg.fractional_matrix_power(Sigma_uu_inv, 0.5)
    G = (Sigma_uu_sqrt_inv.T @ resid_first_no_inst.T @ Xhat @ Sigma_uu_sqrt_inv)/k
    F_cd = np.linalg.eigvalsh(G).min()
    # print('F_cd', F_cd)
    
    pi = beta_first[k_controls:,:]
    pi_vcov_iid = calc_vcov_iid(curr_WZ_gram_inv, resid_first, fs_resid_dof)[k_controls:,k_controls:]
    # print('F_N', 1/k * pi.T @ np.linalg.inv(pi_vcov_iid) @ pi)
    pi_vcov_iid = calc_vcov_iid(curr_WZ_gram_inv, resid_first, N)[k_controls:,k_controls:]
    # print('F_N', 1/k * pi.T @ np.linalg.inv(pi_vcov_iid) @ pi)
    
    # print(fs_resid_dof)
    pi_vcov_hc1 = calc_vcov_hc1(curr_WZ_gram_inv, curr_WZ, resid_first, fs_resid_dof)[k_controls:,k_controls:]
    # print('F_R', 1/k * pi.T @ np.linalg.inv(pi_vcov_hc1) @ pi)
    pi_vcov_hc1 = calc_vcov_hc1(curr_WZ_gram_inv, curr_WZ, resid_first, N)[k_controls:,k_controls:]
    # print('F_R', 1/k * pi.T @ np.linalg.inv(pi_vcov_hc1) @ pi)
    
    WXhat = np.hstack([W, Xhat])
    if fe_params is None: 
        ret_vals = ols_solve(WXhat, y)
        ret_vals += (WXhat, None)
    else:
        ret_vals = fe_solve(WXhat, y, fe_params)
    beta, gram_inv, fitted, resid_second, reg_covariates, y_demeaned = ret_vals
    resid = resid_second - resid_first @ beta[k_controls:,:] 
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