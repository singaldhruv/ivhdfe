import numpy as np
import pandas as pd

# Get sum of all FEs, given value of FE levels 
# Optionally: exclude one FE
def sum_FE(fitted_FE, FE_matrix, groups_FE, levels_FE, c_exclude=None):
    obs_sum_FE = np.zeros((FE_matrix.shape[0], 1))
    c_start_idx = 0
    for c in range(groups_FE):
        c_size = levels_FE[c]
        if c != c_exclude:
            curr_fitted_vals = fitted_FE[c_start_idx:c_start_idx+c_size]
            obs_sum_FE += curr_fitted_vals[FE_matrix[:,c]]
        c_start_idx += c_size
    return obs_sum_FE

# Assumes Gaussian/OLS: optimal FE = demeaned within each level
# Given mu and FE values for all other groups, find optimal FE values for the target FE group
def optimal_single_FE(y, fitted_FE, c_target, FE_matrix, groups_FE, levels_FE, obs_levels_FE): 
    curr_y_demeaned = np.squeeze(y - sum_FE(fitted_FE, FE_matrix, groups_FE, levels_FE, c_target))

    # Calculate the Gaussian optimal FE value for each level of the current group
    level_means = np.bincount(obs_levels_FE[c_target][0], curr_y_demeaned) / obs_levels_FE[c_target][1]
    return level_means[:,np.newaxis]

# Irons-Tuck transition function
# Technically not using the FE fitted values of first group (c=0) in each iteration
def ironstuck_transition(y, prev_fitted_FE, FE_matrix, groups_FE, levels_FE, obs_levels_FE):
    curr_fitted_FE = prev_fitted_FE.copy()
    c_start_idx = 0
    for c in range(groups_FE):
        c_size = levels_FE[c]
        curr_fitted_FE[c_start_idx:c_start_idx+c_size] = optimal_single_FE(y, curr_fitted_FE, c, FE_matrix, groups_FE, levels_FE, obs_levels_FE)
        c_start_idx += c_size
    return curr_fitted_FE

# Irons-Tuck iteration---for faster convergence
def ironstuck_iteration(y, prev_fitted_FE, FE_matrix, groups_FE, levels_FE, obs_levels_FE):
    f = ironstuck_transition(y, prev_fitted_FE, FE_matrix, groups_FE, levels_FE, obs_levels_FE)
    ff = ironstuck_transition(y, f, FE_matrix, groups_FE, levels_FE, obs_levels_FE)
    delta = f - prev_fitted_FE
    # Already converged
    if np.isclose(np.linalg.norm(delta), 0):
        return f
    delta_f = ff - f
    delta_sr = delta_f - delta
    norm_delta_sr = np.linalg.norm(delta_sr)
    dot_delta_f_delta_sr = np.dot(np.squeeze(delta_f), np.squeeze(delta_sr))
    return (ff - (dot_delta_f_delta_sr/norm_delta_sr**2) * delta_f)

# Optimal FE values give y 
# Assumes Gaussian/OLS: (y - X@beta) need not be the summary stat in non-Gaussian cases
# Uses the Irons-Tuck algorithm
def demean_FE(y, FE_matrix, groups_FE, levels_FE, obs_levels_FE, init_fitted_FE, tol=1e-4, max_iter=1000):
    prev_fitted_FE = np.concatenate(init_fitted_FE)
    err = 100
    num_iter = 0
    
    while (err > tol) and (num_iter < max_iter):
        curr_fitted_FE = ironstuck_iteration(y, prev_fitted_FE, FE_matrix, groups_FE, levels_FE, obs_levels_FE)
        err = np.linalg.norm(curr_fitted_FE - prev_fitted_FE)
        prev_fitted_FE = curr_fitted_FE
        num_iter += 1
        
    if num_iter == max_iter:
        print('Warning: Irons-Tuck algorithm did not converge')
    fitted_FE = prev_fitted_FE
    obs_sum_FE = sum_FE(fitted_FE, FE_matrix, groups_FE, levels_FE)
    fitted_FE = np.split(fitted_FE, np.cumsum(levels_FE)[:-1])
    return y - obs_sum_FE