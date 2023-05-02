import pandas as pd
import numpy as np

def make_matrix_psd(matrix):
    if np.ndim(matrix) == 0:
        matrix = np.reshape(matrix, (1,1))
    # Makes a matrix positive definite by clipping negative eigenvalues to 0
    w, a = np.linalg.eigh(matrix)
    if np.any(w < 0):
        matrix = a @ np.diag(np.maximum(w,np.zeros_like(w))) @ np.linalg.inv(a)
    return matrix