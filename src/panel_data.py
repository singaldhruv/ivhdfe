import numpy as np
import pandas as pd

import itertools

class PanelData:
    def __init__(self, orig_df, output=None, endog_vars=[], 
                instruments=[], controls=[], 
                fixed_effects=[], se_clusters=[],
                skip_constant=False):
        df = orig_df
        assert isinstance(df, pd.DataFrame), "df must be a Pandas DataFrame"
        # Check continuous columns
        if instruments == []:
            cts_col_types = ['covariate', 'control']
            cts_cols = [endog_vars, controls]
        else:
            cts_col_types = ['covariate', 'instrument', 'control']
            cts_cols = [endog_vars, instruments, controls]
            
        df, cts_cols, dup_cols = check_cts_cols(df, output, cts_cols, cts_col_types)
        for col in dup_cols:
            if col[1] == 'covariate':
                endog_vars.remove(col[0])
            elif col[1] == 'instrument':
                instruments.remove(col[0])
            else:
                controls.remove(col[0])
        
        # Check categorical columns and store univariate levels 
        df, uni_cat_cols, fe_cols, clust_cols = check_cat_cols(df, fixed_effects, se_clusters)
        self.uni_cat_cols = uni_cat_cols
        
        
        # Subset to useful columns
        # Drop observations with NA or Inf in any of the useful columns
        useful_cols = [output] + cts_cols + uni_cat_cols
        df = df[useful_cols].copy()
        df = df.replace([np.inf,-np.inf], np.nan).dropna().reset_index(drop=True)
        
        # Add a constant if needed 
        if (fe_cols == []) and (skip_constant == False):
            df['_constant'] = 1
            cts_cols = ['_constant'] + cts_cols
            
        # Iterate for checking collinearity in cts columns and 
        # Singleton in FE until hitting a fixed point
        any_change = True
        while any_change:
            any_change = False
            prev_size = df.shape[0]
            df, cts_cols = check_collinear(df, cts_cols)
            df, clust_column_labels = create_cat_levels(df, clust_cols, 'clust', remove_singletons=False)
            df, fe_column_labels = create_cat_levels(df, fe_cols, 'fe', remove_singletons=True)
            if df.shape[0] < prev_size:
                any_change = True
        
        df = df.reset_index(drop=True)
        self.df = df
        self.N = df.shape[0]
        
        self.cts_cols = cts_cols
        self.output = output
        
        # y is the outcome variable
        self.y = df[[output]].values
        
        # X is the matrix of main endogenous covariates
        self.endog_vars = [col for col in endog_vars if col in cts_cols]
        self.X = df[self.endog_vars].values
        
        # W is the matrix of controls (included instruments)
        self.controls = [col for col in controls if col in cts_cols]
        if (fe_cols == []) and (skip_constant == False):
            # Add constant to included controls, if used
            self.controls = ['_constant'] + self.controls
        self.W = df[self.controls].values 
        self.resid_dof = self.N - (len(self.endog_vars) + len(self.controls))
        
        # Z is the matrix of excluded instruments
        self.instruments = [col for col in instruments if col in cts_cols]
        self.Z = df[self.instruments].values
        if len(self.instruments) > 0:
            assert len(self.instruments) >= len(self.endog_vars), 'Under-identified specification'
        
        self.initialize_fixed_effects(fe_column_labels)
        self.initialize_clusters(clust_column_labels)
        
    #################
    # Fixed effects #
    ################# 
    def initialize_fixed_effects(self, fe_column_labels):
        # This stores the FE column names
        self.fixed_effects = fe_column_labels
        # This stores the "actual" categorical values for each FE column, which can be mapped to the univariate levels
        
        # Terminology: 
        # (group === column)
        # (level == value)
        
        # FE is a matrix of categorical covariates
        # Values in FE are zero-indexed and continguous 
        # (i.e., for class c: 0, 1, 2, 3, ..., #levels-1)
        self.FE_matrix = self.df[self.fixed_effects].values
        
        # Number of FE groups
        self.groups_FE = np.int64(self.FE_matrix.shape[1])
        # Number of non-NA levels per FE group (NA is coded as -1 in each FE group)
        self.levels_FE = np.max(self.FE_matrix, axis=0) + 1
        
        # Number of identified FE levels (degrees of freedom lost)
        self.N_FE = self.levels_FE.sum()

        # Remove redundant FE levels and adjust N_FE
        # Remove partitioned FEs and adjust N_FE
        # For each FE group, store the partitioned FE levels (i.e., can be normalized to zero wlog)
        # Also store the group and levels which partition these coarse levels
        self.partitioned_FE = [dict() for c in range(self.groups_FE)]
        for (coarse_fe, fine_fe) in itertools.permutations(range(self.FE_matrix.shape[1]), 2):
            curr_partition_df = find_partitioned_levels(self.FE_matrix[:,fine_fe][:,np.newaxis], self.FE_matrix[:,coarse_fe][:,np.newaxis])
            curr_partition_df = curr_partition_df.groupby('coarse').agg({'fine': (lambda x: (fine_fe, list(x)))}).reset_index()
            curr_partitioned_levels = curr_partition_df['coarse'].values
            self.N_FE -= len(curr_partitioned_levels)
            for coarse_level in curr_partitioned_levels:
                self.partitioned_FE[coarse_fe][coarse_level] = curr_partition_df[curr_partition_df['coarse'] == coarse_level]['fine'].values[0]
            
        # One arbitrary non-partitioned level of each FE group (after the first FE group) is redundant
        self.redundant_FE = np.zeros((self.groups_FE,)).astype(int)
        for c in range(1, self.groups_FE): 
            c_partitioned_levels = self.partitioned_FE[c].keys()
            if (len(c_partitioned_levels) < self.levels_FE[c]):
                # Only remove one level if 
                # All the FE levels are not partitioned
                self.redundant_FE[c] = 1
        self.N_FE -= self.redundant_FE.sum()
        
        temp_fe_groups = list(range(self.groups_FE))
        removed_so_far = 0
        for c in temp_fe_groups:
            c_idx = c - removed_so_far
            # If partitioned levels + redundant levels == all levels:
            # Remove the FE group
            c_removed_levels = len(self.partitioned_FE[c_idx].keys())
            c_removed_levels += self.redundant_FE[c_idx]
            if c_removed_levels == self.levels_FE[c_idx]:
                del self.fixed_effects[c_idx]
                self.FE_matrix = np.delete(self.FE_matrix, c_idx, axis=1)
                self.groups_FE = self.FE_matrix.shape[1]
                self.levels_FE = np.max(self.FE_matrix, axis=0) + 1
                self.redundant_FE = np.delete(self.redundant_FE, c_idx)
                self.partitioned_FE = np.delete(self.partitioned_FE, c_idx)
                # self.N_FE has already been adjusted
                removed_so_far += 1
        
        # Remove degrees of freedom from the model 
        self.resid_dof -= self.N_FE
            
        # Indices and count of observations in each FE group and level
        self.obs_levels_FE = []
        for c in range(self.groups_FE):
            _, group_obs_idx, group_level_counts = np.unique(self.FE_matrix[:, c], return_inverse=True, return_counts=True) 
            self.obs_levels_FE.append((group_obs_idx, group_level_counts))
        
        self.init_fitted_FE = list((map(lambda x: np.zeros((x,1)), self.levels_FE)))
        
    ###############
    # SE clusters #
    ############### 
    def initialize_clusters(self, clust_column_labels):
        self.se_clusters = clust_column_labels
        self.clust_df = self.df[self.se_clusters].copy()
        
        if (len(self.se_clusters) > 0) and (self.N_FE > 0):
            # Add back the degrees of freedom to the model for non-partitioned FE levels
            # i.e., FE levels which are nested within some cluster
            nested_FE = [set([]) for _ in range(self.groups_FE)]
            partitioned_FE = [set(self.partitioned_FE[fe_c].keys()) for fe_c in range(self.groups_FE)]
            for clust_c in self.clust_df.columns:
                clust_array = self.clust_df[[clust_c]].copy()
                for fe_c in range(self.groups_FE):
                    nest_df = find_nested_levels(self.FE_matrix[:,fe_c][:,np.newaxis], clust_array)
                    curr_nested_fe = set(nest_df['fine'].unique())
                    nested_FE[fe_c] = nested_FE[fe_c] | curr_nested_fe
            nested_dof = 0
            for fe_c in range(self.groups_FE):
                fe_c_nested_dof = len(nested_FE[fe_c] - partitioned_FE[fe_c])
                fe_c_non_partitioned_levels = set(np.arange(self.levels_FE[fe_c])) - partitioned_FE[fe_c]
                nested_dof += fe_c_nested_dof
                if fe_c_nested_dof == len(fe_c_non_partitioned_levels):
                    # All non-partitioned levels are nested within some cluster
                    # Subtract the degrees of freedom for the redundant levels
                    nested_dof -= self.redundant_FE[fe_c]
            # If all FE levels are nested within some cluster, remove the degrees of freedom for the intercept
            nested_dof -= 1 if (nested_dof == self.N_FE) else 0
            self.nested_resid_dof = nested_dof
            self.resid_dof += nested_dof
        else:
            self.nested_resid_dof = 0
            
##################################
# Helpers for continuous columns #
##################################
def check_cts_cols(df, output, combined_cts_cols, cts_col_types):
    assert isinstance(output, str), f"Output {output} is not a string"
    assert (output in df.columns), f"Output {output} is not a data column"
    cts_cols = []
    dup_cols = []
    for curr_cts_cols, curr_cols_type in zip(combined_cts_cols, cts_col_types):
        assert (isinstance(curr_cts_cols, list)), f"{curr_cols_type} {curr_cts_cols} is not a list"
        for col in curr_cts_cols:
            assert isinstance(col, str), f"In {curr_cols_type} {curr_cts_cols}, {col} is not a list"
            assert (col in df.columns), f"In {curr_cols_type} {curr_cts_cols}, {col} is not a data column" 
            assert col != output, f"In {curr_cols_type} {curr_cts_cols}, {col} is the output"
            assert (np.issubdtype(df[col].dtype, np.number)), f"In {curr_cols_type} {curr_cts_cols}, {col} is not numeric"
            if col not in cts_cols:
                cts_cols.append(col)
            else:
                print(f'Skipping duplicated {curr_cols_type}: {col}')
                dup_cols.append((col, curr_cols_type))
    return df, cts_cols, dup_cols

def check_collinear(df, cts_cols):
    orthogonal_cols = [cts_cols[0]]
    collinear_cols = []
    full_rank_matrix = df[[cts_cols[0]]].values
    
    for col in cts_cols[1:]:
        temp_matrix = np.hstack((full_rank_matrix, df[[col]].values))
        if np.linalg.matrix_rank(temp_matrix) < temp_matrix.shape[1]:
            collinear_cols.append(col)
        else:
            orthogonal_cols.append(col)
            full_rank_matrix = temp_matrix
            
    if collinear_cols != []: 
        print(f'Skipping collinear: {collinear_cols}')
    return df, orthogonal_cols

###################################
# Helpers for categorical columns #
###################################
def check_cat_cols(df, fixed_effects, se_clusters):
    uni_cat_cols = []
    
    fe_cols = []
    assert isinstance(fixed_effects, list), f"Fixed effects {fixed_effects} is not a list"
    for curr_fe_cols in fixed_effects:
        if not isinstance(curr_fe_cols, tuple):
            assert isinstance(curr_fe_cols, str), f"Single FE {curr_fe_cols} is not a string"
            curr_fe_cols = (curr_fe_cols,)
        curr_fe_cols = list(curr_fe_cols)
        for col in curr_fe_cols:
            assert isinstance(col, str), f"{col} in the FE {curr_fe_cols} is not a string"
            assert (col in df.columns), f"{col} in the FE {curr_fe_cols} is not a data column"
            if col not in uni_cat_cols:
                uni_cat_cols.append(col)
        if fe_cols not in fe_cols:
            fe_cols.append(curr_fe_cols)
        else:
            print(f'Skipping duplicate FE: {curr_fe_cols}')
        
    clust_cols = [] 
    assert isinstance(se_clusters, list), f"SE clusters {se_clusters} is not a list"
    for curr_clust_cols in se_clusters:
        if not isinstance(curr_clust_cols, tuple):
            assert isinstance(curr_clust_cols, str), f"Single cluster {curr_clust_cols} is not a string"
            curr_clust_cols = (curr_clust_cols,)
        curr_clust_cols = list(curr_clust_cols)
        for col in curr_clust_cols:
            assert isinstance(col, str), f"{col} in the cluster {curr_clust_cols} is not a string"
            assert (col in df.columns), f"{col} in the cluster {curr_clust_cols} is not a data column"
            if col not in uni_cat_cols:
                uni_cat_cols.append(col)
        if curr_clust_cols not in clust_cols:
            clust_cols.append(curr_clust_cols)
        else:
            print(f'Skipping duplicate cluster: {curr_clust_cols}')
        
    for col in uni_cat_cols:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
        
    return df, uni_cat_cols, fe_cols, clust_cols

# Find levels for the categorical column fine_col which
# partition some levels in categorical column coarse_col, that is:
# There are two conditions: (x1, x2, ..., xp) in fine_col
# partition y1 in coarse_col iff:
# 1. each of (x1, x2, ..., xp) only appears with y1
# 2. y1 only appears with each of (x1, x2, ..., xp)
def find_partitioned_levels(fine_col, coarse_col):
    partition_df = pd.DataFrame(np.hstack([fine_col, coarse_col]), columns=['fine','coarse']).drop_duplicates()
    # Get closure of each level in coarser_col under coarse_col
    closure_df = partition_df.rename({'coarse': 'closure'}, axis=1).merge(partition_df, on='fine', how='left').drop('fine', axis=1).drop_duplicates()
    coarse_counts = closure_df['coarse'].value_counts()
    # If a level in coarse_col only has one element in its closure, it is partitioned
    partitioned_coarse = coarse_counts[coarse_counts == 1].index
    partition_df = partition_df.loc[partition_df['coarse'].isin(partitioned_coarse), :].reset_index(drop=True)
    return partition_df

# Find levels for the categorical column fine_col which are 
# nested within levels in categorical column coarse_col, that is:
# There is one condition: (x1, x2, ..., xp) in fine_col
# are nested in y1 in coarse_col iff:
# each of (x1, x2, ..., xp) only appears with y1
def find_nested_levels(fine_col, coarse_col):
    nest_df = pd.DataFrame(np.hstack([fine_col, coarse_col]), columns=['fine','coarse']).drop_duplicates()
    fine_counts = nest_df['fine'].value_counts()
    nested_fine = fine_counts[fine_counts == 1].index
    # If a level in fine_col appears with only one level of coarse_col, it is nested
    nest_df = nest_df.loc[nest_df['fine'].isin(nested_fine), :].reset_index(drop=True)
    return nest_df

# Factorizes a column or list of columns 
# curr_cat can be a tuple specifying interaction of categorical variables
def factorize(df, curr_cat, cat_tag):
    curr_cat_label = f"{cat_tag}_{'#'.join(curr_cat)}"
    df[curr_cat_label] = np.zeros(df.shape[0], dtype=np.dtype('uint64'))
    possible_max = np.iinfo(np.dtype('uint64')).max
    running_max = 1
    for col in curr_cat:
        col_max = df[col].max()
        running_max *= col_max
        assert running_max < possible_max, f"Interacted levels in {curr_cat} are too numerous"
        df[curr_cat_label] = df[curr_cat_label]*col_max+ df[col]
    codes, _ = pd.factorize(df[curr_cat_label])
    df[curr_cat_label] = codes
    return df, curr_cat_label

def create_cat_levels(df, curr_cat_cols, cat_tag='cat', remove_singletons=True):
    cat_labels = []
    for curr_cat in curr_cat_cols:
        df, curr_cat_label = factorize(df, curr_cat, cat_tag)
        
        # Remove singletons and refactorize
        if remove_singletons: 
            curr_counts = df[curr_cat_label].value_counts()
            non_singletons = curr_counts[curr_counts > 1].index
            df = df.loc[df[curr_cat_label].isin(non_singletons), :].reset_index(drop=True)
            df, curr_cat_label = factorize(df, curr_cat, cat_tag)

        cat_labels.append(curr_cat_label)
    return df, cat_labels