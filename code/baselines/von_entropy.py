""" Calculate Von Neumann entropy """
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pennylane.math.quantum import dm_from_state_vector, vn_entropy


def quantum_von_neumann_p(enc_data):
    # Von Neumann entropy
    mutual_info = []
    mutual_info = [Parallel(n_jobs=-1)(delayed(vn_entropy)(np.outer(enc, enc.conj()), indices=[0], base=2) \
                for enc in  enc_data)]
    average_mutual = np.mean(mutual_info)
    # print('average_mutual: ', average_mutual)
    return average_mutual


def is_normalized(rho):
    trace = np.trace(rho)
    return np.isclose(trace, 1.0)

def is_positive_semidefinite(matrix):
    return np.all(np.linalg.eigvals(matrix) >= 0)

def is_hermitian(matrix):
    return np.allclose(matrix, matrix.conj().T)

def is_trace_one(matrix):
    return np.isclose(np.trace(matrix), 1.0)

def regularize_matrix(matrix, epsilon=1e-8):
    identity = np.identity(matrix.shape[0])
    regularized_matrix = matrix + epsilon * identity
    return regularized_matrix

def normalize_matrix(matrix):
    trace = np.trace(matrix)
    normalized_matrix = matrix / trace
    return normalized_matrix
