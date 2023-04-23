import numpy as np
import scipy as sp
import pandas as pd

from fixed_effects import demean_FE

# Use QR decomposition to first get Xq @ Xr = X
# If no y is provided, just returns (X'X)^-1, (X' @ X)^-1 @ X'
# If y is provided, solve OLS: y = X @ beta + e
def ols_solve(X, y=None):
    Xq, Xr = np.linalg.qr(X)
    gram_inv = np.linalg.solve(Xr.T @ Xr, np.eye(X.shape[1]))
    mp_inv = gram_inv @ X.T
    if y is None:
        return gram_inv, mp_inv
    beta = sp.linalg.solve_triangular(Xr, Xq.T @ y)
    fitted = X @ beta 
    resid = y - fitted
    return beta, gram_inv, fitted, resid

# First stage: X = [W, Z] @ alpha + u
# Second stage: y = [W, Xhat] @ beta + e
def tsls_solve(X, W, Z, y, fe_params=None):
    WZ = np.hstack([W, Z])
    Xhat = np.zeros_like(X)
    resid_first = np.zeros_like(X)
    f_stats = np.zeros((X.shape[1],1))
    
    for x_idx in range(X.shape[1]):
        curr_X = X[:,x_idx][:,np.newaxis]
        if fe_params is None:
            ret_vals = ols_solve(WZ, curr_X)
            ret_vals = ret_vals + (None, None)
            
            ret_vals_no_inst = ols_solve(W, curr_X)
            ret_vals_no_inst = ret_vals_no_inst + (None, None)
        else:
            ret_vals = fe_solve(WZ, curr_X, fe_params)
            ret_vals_no_inst = fe_solve(W, curr_X, fe_params)
        _, _, curr_Xhat, curr_resid_first, _, _ = ret_vals
        Xhat[:,x_idx] = np.squeeze(curr_Xhat)
        resid_first[:,x_idx] = np.squeeze(curr_resid_first)
        
        _, _, _, curr_resid_first_no_inst, _, _ = ret_vals_no_inst
        curr_ssr_first = np.sum(curr_resid_first**2)
        curr_ssr_first_no_inst = np.sum(curr_resid_first_no_inst**2)
        curr_f_stat = (curr_ssr_first_no_inst - curr_ssr_first)  / curr_ssr_first
        f_stats[x_idx] = curr_f_stat
        
    WXhat = np.hstack([W, Xhat])
    if fe_params is None: 
        ret_vals = ols_solve(WXhat, y)
        ret_vals += (WXhat, None)
    else:
        ret_vals = fe_solve(WXhat, y, fe_params)
    beta, gram_inv, fitted, resid_second, reg_covariates, y_demeaned = ret_vals
    resid = resid_second - resid_first @ beta[W.shape[1]:,:] 
    return beta, gram_inv, fitted, resid, reg_covariates, y_demeaned, f_stats

# Assumes Gaussian/OLS: Solve OLS using demeaned X and y
def fe_solve(X, y, fe_params):
    y_demeaned = demean_FE(y, fe_params['FE_matrix'], fe_params['groups_FE'], fe_params['levels_FE'], fe_params['obs_levels_FE'], fe_params['init_fitted_FE'])
    
    # Can probably (?) do this using matrices
    # However, it is efficient enough
    X_demeaned = np.zeros_like(X)
    for k in range(X.shape[1]):
        curr_X = X[:,k][:,np.newaxis]
        curr_X_demeaned = demean_FE(curr_X, fe_params['FE_matrix'], fe_params['groups_FE'], fe_params['levels_FE'], fe_params['obs_levels_FE'], fe_params['init_fitted_FE'])
        X_demeaned[:,k] = np.squeeze(curr_X_demeaned)
        
    beta, gram_inv, fitted, resid = ols_solve(X_demeaned, y_demeaned)
    return beta, gram_inv, fitted, resid, X_demeaned, y_demeaned