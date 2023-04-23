import numpy as np
import pandas as pd

import itertools

from panel_data import factorize

def calc_std(vcov):
    if np.ndim(vcov) == 0:
        vcov = np.reshape(vcov, (1,1))
    # Makes a vcov positive definite by clipping negative eigenvalues to 0
    w, a = np.linalg.eig(vcov)
    if np.any(w < 0):
        vcov = a @ np.diag(np.maximum(w,np.zeros_like(w))) @ np.linalg.inv(a)
    std = np.sqrt(np.diag(vcov))
    std = np.reshape(std, (len(std),1))
    return std

def calc_vcov_iid(gram_inv, rss, resid_dof):
    return (rss/resid_dof) * gram_inv

# Bottleneck: Loop is slow (still way faster than matrix multiplication way)
def calc_vcov_hc1(gram_inv, reg_covariates, sr_resid, resid_dof):
    N = sr_resid.shape[0]
    K = reg_covariates.shape[1]
    
    vcov_hc0 = np.zeros((K,K)) 
    for n in range(N):
        vcov_hc0 += sr_resid[n,0] * reg_covariates[n,:][:,np.newaxis] @ reg_covariates[n,:][np.newaxis,:]
    
    vcov_hc0 = gram_inv @ vcov_hc0 @ gram_inv.T
    vcov_hc1 = (N/resid_dof) * vcov_hc0
    return vcov_hc1


# Bottleneck: Finding uniques and indices for them
def calc_meat_oneway(reg_covariates, resid, clust_df, oneway_dof_adj=True):
    moments = np.multiply(reg_covariates, resid)
    
    moments_df, clust_col, _ = factorize(clust_df.copy(), clust_df.columns, 'oneway')
    moments_cols = []
    for k in range(moments.shape[1]):
        moments_df[f'moments_{k}'] = moments[:,k]
        moments_cols.append(f'moments_{k}')
    moments_sum = moments_df.groupby(clust_col)[moments_cols].sum().values.T[:,:,np.newaxis]
    M = moments_sum.shape[1]
    meat = np.einsum('jmo,kmo', moments_sum, moments_sum)

    if oneway_dof_adj:
        meat = (M/(M-1)) * meat
    return meat, M
    
def calc_vcov_clust_multiway(bread, reg_covariates, resid, resid_dof, clust_df, clust_dof_adj='min'):
    N = reg_covariates.shape[0]
    K = reg_covariates.shape[1]
    dof_adj_factor = ((N-1)/(resid_dof))
    
    assert clust_dof_adj in ['min', 'conventional']
    oneway_dof_adj = (clust_dof_adj == 'conventional')
    all_clust = clust_df.columns
    meat = np.zeros((K, K))
    clust_Ms = []
    for interact_n in range(1, len(all_clust)+1):
        for curr_clust in itertools.combinations(all_clust, interact_n):
            curr_meat, curr_clust_M = calc_meat_oneway(reg_covariates, resid, clust_df[list(curr_clust)].copy(), oneway_dof_adj)
            meat += ((-1)**(interact_n+1)) * curr_meat
            if interact_n == 1:
                clust_Ms.append(curr_clust_M)
                
    vcov_clust_multiway = dof_adj_factor * (bread @ meat @ bread)
    if clust_dof_adj == 'min':
        M_min = min(clust_Ms)
        vcov_clust_multiway = (M_min/(M_min-1)) * vcov_clust_multiway
        
    return vcov_clust_multiway