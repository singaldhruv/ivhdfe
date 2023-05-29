import numpy as np
import pandas as pd

import itertools

from .panel_data import factorize
from .linalg_helpers import make_matrix_psd

def calc_vcov(gram_inv, reg_covariates, resid, resid_dof, vcov_params=None):
    if vcov_params is None:
        vcov_params = {'type': 'iid'}
        
    assert np.ndim(resid) == 2, 'resid must be a 2-D array'
    assert resid.shape[1] in [1,2], 'resid must have 1 or 2 columns'
    if resid.shape[1] == 1:
        resid_alt = resid.copy()
        vcov_params['single_resid'] = True
    else:
        vcov_params['single_resid'] = False
        resid_alt = resid[:,1][:,np.newaxis]
        resid = resid[:,0][:,np.newaxis]
        
    if 'clust_df' in vcov_params:
        assert 'clust_dof_adj' in vcov_params, 'clust_dof_adj must be specified if clust_df is specified'
        return calc_vcov_clust_multiway(gram_inv, reg_covariates, resid, resid_alt, resid_dof, vcov_params)
    elif vcov_params['vcov_type'] == 'iid':
        return calc_vcov_iid(gram_inv, resid, resid_alt, resid_dof)
    elif vcov_params['vcov_type'] == 'robust':
        return calc_vcov_hc1(gram_inv, reg_covariates, resid, resid_alt, resid_dof)
    else:
        raise ValueError('If clusters are not specified, vcov_type must be iid or robust') 
        
def calc_std(vcov):
    std = np.sqrt(np.diag(vcov))
    std = np.reshape(std, (len(std),1))
    return std

def calc_vcov_iid(gram_inv, resid, resid_alt, resid_dof):
    sr_resid = np.multiply(resid, resid_alt)
    return make_matrix_psd((np.sum(sr_resid)/resid_dof) * gram_inv)

def calc_vcov_hc1(gram_inv, reg_covariates, resid, resid_alt, resid_dof):
    N = resid.shape[0]
    K = reg_covariates.shape[1]
    sr_resid = np.multiply(resid, resid_alt)
    
    vcov_hc0 = np.zeros((K,K)) 
    for n in range(N):
        vcov_hc0 += sr_resid[n,0] * reg_covariates[n,:][:,np.newaxis] @ reg_covariates[n,:][np.newaxis,:]
    
    vcov_hc0 = gram_inv @ vcov_hc0 @ gram_inv.T
    vcov_hc1 = (N/resid_dof) * vcov_hc0
    return make_matrix_psd(vcov_hc1)

def calc_meat_oneway(reg_covariates, resid, resid_alt, clust_df, clust_params):
    moments = np.multiply(reg_covariates, resid)
    
    moments_df, clust_col = factorize(clust_df.copy(), clust_df.columns, 'oneway')
    moments_cols = []
    for k in range(moments.shape[1]):
        moments_df[f'moments_{k}'] = moments[:,k]
        moments_cols.append(f'moments_{k}')
    moments_sum = moments_df.groupby(clust_col)[moments_cols].sum().values.T[:,:,np.newaxis]
    M = moments_sum.shape[1]
    
    if clust_params['single_resid']:
        moments_alt_sum = moments_sum
    else:
        moments_alt = np.multiply(reg_covariates, resid_alt)
        for k in range(moments.shape[1]):
            moments_df[f'moments_{k}'] = moments_alt[:,k]
        moments_alt_sum = moments_df.groupby(clust_col)[moments_cols].sum().values.T[:,:,np.newaxis]
    meat = np.einsum('jmo,kmo', moments_sum, moments_alt_sum)

    if clust_params['clust_dof_adj'] == 'conventional':
        meat = (M/(M-1)) * meat
    return meat, M
    
def calc_vcov_clust_multiway(bread, reg_covariates, resid, resid_alt, resid_dof, clust_params):
    N = reg_covariates.shape[0]
    K = reg_covariates.shape[1]
    dof_adj_factor = ((N-1)/(resid_dof))
    
    clust_df = clust_params['clust_df']
    assert clust_params['clust_dof_adj'] in ['min', 'conventional']
    all_clust = clust_df.columns
    meat = np.zeros((K, K))
    clust_Ms = []
    for interact_n in range(1, len(all_clust)+1):
        for curr_clust in itertools.combinations(all_clust, interact_n):
            curr_meat, curr_clust_M = calc_meat_oneway(reg_covariates, resid, resid_alt, clust_df[list(curr_clust)].copy(), clust_params)
            meat += ((-1)**(interact_n+1)) * curr_meat
            if interact_n == 1:
                clust_Ms.append(curr_clust_M)
                
    vcov_clust_multiway = dof_adj_factor * (bread @ meat @ bread)
    if clust_params['clust_dof_adj'] == 'min':
        M_min = min(clust_Ms)
        vcov_clust_multiway = (M_min/(M_min-1)) * vcov_clust_multiway
    return make_matrix_psd(vcov_clust_multiway)