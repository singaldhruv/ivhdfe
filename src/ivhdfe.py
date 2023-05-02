import pandas as pd
import numpy as np

from panel_data import PanelData
from solvers import tsls_solve, fe_solve
from vcov import calc_vcov, calc_std

def ivhdfe(df, output, 
        endog_vars=[],
        instruments=[],
        controls=[],
        fixed_effects=[],
        se_clusters=[],
        vcov_type='iid',
        skip_constant=False,
        verbose=True):
    
    panel = PanelData(df,
        output=output, 
        endog_vars=endog_vars,
        instruments=instruments,
        controls=controls,
        fixed_effects=fixed_effects,
        se_clusters=se_clusters,
        skip_constant=skip_constant,
        verbose=verbose)
    
    N = panel.N
    X = panel.X
    W = panel.W
    Z = panel.Z
    WX = np.hstack([W, X])

    y = panel.y
    yss = np.sum(np.power(y - np.mean(y), 2))
    yss_dof = N - 1
    
    resid_dof = panel.resid_dof
    
    if len(panel.se_clusters) > 0:
        # Overrides any value for se argument
        vcov_params = {
            'clust_df': panel.clust_df,
            'clust_dof_adj': 'min'
        }
        if verbose:
            print('Standard errors are clustered')
    elif vcov_type == 'iid':
        vcov_params = {'vcov_type': 'iid'}
        if verbose:
            print('Standard errors assume homoskedasticity (iid)')
    elif vcov_type == 'robust':
        vcov_params = {'vcov_type': 'robust'}
        if verbose:
            print('Standard errors assume heteroskedasticity (HC1)')
    else:
        raise ValueError('If clusters are not specified, se must be iid or robust') 

    if (panel.fixed_effects != []):
        fe_params = {
            'FE_matrix': panel.FE_matrix,
            'groups_FE': panel.groups_FE,
            'levels_FE': panel.levels_FE,
            'obs_levels_FE': panel.obs_levels_FE,
            'init_fitted_FE': panel.init_fitted_FE,
            'N_FE': panel.N_FE,
            'nested_resid_dof': panel.nested_resid_dof,
        }
    else:
        fe_params = None
        
    if (panel.instruments != []):
        # Testing for weak instruments needs these
        iv_params = {
            'vcov_params': vcov_params,
            'resid_dof': resid_dof,
        }
        ret_vals = tsls_solve(X, W, Z, y, iv_params, fe_params)
        f_stats = ret_vals[-1]
        ret_vals = ret_vals[:-1]
    else:
        f_stats = None
        ret_vals = fe_solve(WX, y, fe_params)
    
    beta, gram_inv, fitted, resid, reg_covariates, y_demeaned = ret_vals
    sr_resid = np.power(resid,2)
    rss = np.sum(sr_resid)
    
    if fe_params is not None:
        assert y_demeaned is not None, "Output variable was not demeaned properly"
        adj_resid_dof = resid_dof - fe_params['nested_resid_dof']
        wss_dof = N - fe_params['N_FE']
        wss = np.sum(np.power(y_demeaned, 2))
        within_r2 = 1 - rss/wss
        adj_within_r2 = 1 - (rss/adj_resid_dof)/(wss/wss_dof)
    else:
        adj_resid_dof = resid_dof
        wss_dof = None
        wss = None
        within_r2 = None
        adj_within_r2 = None
    
    if ('_constant' not in panel.controls) and (fe_params is None):
        yss = np.sum(np.power(y, 2))
        yss_dof = N
        
    sigma2 = (rss/resid_dof) 
    r2 = 1 - rss/yss
    adj_r2 = 1 - (rss/(adj_resid_dof))/(yss/yss_dof)
    
    if beta is None:
        vcov = None
        std = None
        covariates = None
    else:
        vcov = calc_vcov(gram_inv, reg_covariates, resid, resid_dof, vcov_params)
        std = calc_std(vcov)

        # Reorder output coefficients and standard errors
        num_controls = len(panel.controls) 
        num_endog_vars = len(panel.endog_vars)
        idxr = list(range(num_controls, num_controls + num_endog_vars))
        covariates = panel.endog_vars
        if num_controls > 0:
            if panel.controls[0] == '_constant':
                idxr = [0] + idxr + list(range(1, num_controls))
                covariates = ['_constant'] + covariates + panel.controls[1:]
            else:
                idxr = idxr + list(range(num_controls))
                covariates = covariates + panel.controls
        beta = beta[idxr,:].T
        std = std[idxr,:].T
    
    return {
        'nobs': N,
        'covariates': covariates,
        'beta': beta,
        'se': std,
        'f_stats': f_stats,
        'r2': r2,
        'adj_r2': adj_r2,
        'within_r2': within_r2,
        'adj_within_r2': adj_within_r2,
        'sigma2': sigma2,
        'fitted': fitted,
        'resid': resid,
    }