import os
import time
import numpy as np
from tqdm import tqdm

from qiskit.quantum_info import partial_trace

__all__ = ['trace_distance', 'calculate_mean_distance', \
           'calculate_mean_distance_to_max_mixed_state', 'calculate_entanglement', \
           'calculate_expressibility', 'compute_rank_consistency']

def trace_distance(state1, state2):
    """
    Calculate the trace distance between two quantum states.
    state1: First quantum state vector.
    state2: Second quantum state vector.
    """
    # Calculate density matrices
    rho1 = np.outer(state1, state1.conj())
    rho2 = np.outer(state2, state2.conj())

    # Calculate trace distance using partial trace
    trace_distance = 0.5 * np.linalg.norm(partial_trace(rho1 - rho2, [0]))
    return trace_distance

def calculate_mean_distance(data1, data2):
    """
    Calculate the mean distance between two data sets.
    data1: First data set with shape (n, d).
    data2: Second data set with shape (m, d).
    """
    distance = np.zeros((data1.shape[0], data2.shape[0]))
    for i in range(data1.shape[0]):
        for j in range(data2.shape[0]):
            distance[i, j] = trace_distance(data1[i], data2[j])
    mean_distance = distance.mean()
    return mean_distance

def calculate_mean_distance_to_max_mixed_state(data):
    """
    Calculate the mean distance of a data set to the maximally mixed state.
    data: Data set with shape (n, d).
    """
    max_mixed_state = np.ones(data.shape[1]) / np.sqrt(data.shape[1])
    distance = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        distance[i] = trace_distance(data[i], max_mixed_state)
    mean_distance = distance.mean()
    return mean_distance

def calculate_entanglement(data):
    """
    Calculate the entanglement among the data points.
    data: Data set with shape (n, d).
    """
    entanglement = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            entanglement += trace_distance(data[i], data[j])
    entanglement /= data.shape[0] ** 2
    return entanglement

def calculate_expressibility(data):
    """
    Calculate the expressibility among the data points.
    data: Data set with shape (n, d).
    """
    expressibility = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            expressibility += trace_distance(data[i], data[j])
    expressibility /= data.shape[0] ** 2
    return expressibility

def compute_rank_consistency(array1, array2):
    """
    Compare the rank consistency between two arrays.
    """
    array1_sorted_indices = sorted(range(len(array1)), key=lambda k: array1[k], reverse=True)
    array2_sorted_indices = sorted(range(len(array2)), key=lambda k: array2[k], reverse=True)

    score = 0
    for i in range(len(array1)):
        if array1_sorted_indices[i] == array2_sorted_indices[i]:
            score += 1

    return score