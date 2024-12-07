import itertools
import numpy as np
from scipy import linalg
from joblib import Parallel, delayed
from qiskit import quantum_info
from qiskit.quantum_info import Statevector, partial_trace

__all__ = ['_random_unitary', '_haar_integral', 'calculate_expressibility',\
           '_cal_partial_trace', 'calculate_entangling']

""" Expressibility Capability =  Harr_integral - PQC_integral """
def _random_unitary(N_dims):
    """ 
        Return a Haar distributed random unitary from U(N_dims)
    """
    Z = np.random.randn(N_dims, N_dims) + 1.0j * np.random.randn(N_dims, N_dims)
    # QR matrix decomposition, Q is orthogonal, R is Triangular matrix
    [Q, R] = linalg.qr(Z)
    D = np.diag(np.diagonal(R) / np.abs(np.diagonal(R)))

    return np.dot(Q, D)


def _haar_integral(num_qubits, samples=2048):
    N_dims = 2**num_qubits
    randunit_density = np.zeros((N_dims, N_dims), dtype=complex)

    zero_state = np.zeros(N_dims, dtype=complex)
    zero_state[0] = 1

    for _ in range(samples):
        A = np.matmul(zero_state, _random_unitary(N_dims)).reshape(-1, 1)
        randunit_density += np.matmul(A, A.conj().T)
    
    randunit_density /= samples

    return randunit_density


def calculate_expressibility(data, samples=2048):
    """ 
        Returns the expressibility measure for the given quantum states.
    """
    reshape_data = data.reshape(data.shape[0], -1)
    d1, d2 = reshape_data.shape # [bsz, 2**num_qubits]
    num_qubits = int(np.log2(d2))
    haar_density_matrix = _haar_integral(num_qubits, samples)
    pqc_density_matrix = np.reshape(reshape_data, [d1, d2, 1]) @ np.reshape(reshape_data, [d1, 1, d2])

    expressibility = np.linalg.norm(haar_density_matrix - pqc_density_matrix)
    
    return expressibility

""" 
    Entangling Capability 
"""
def _cal_partial_trace(state):
    """ 
        Calculate partial trace of a quantum state.
    """
    num_qubits = len(state.dims())
    qb = list(range(num_qubits))
    qubits_list = list(itertools.combinations(qb, num_qubits-1))
    entropy = 0.0

    for qubits_combination in qubits_list:
        dens = quantum_info.partial_trace(state, qubits_combination).data
        trace = np.trace(dens ** 2)
        entropy += trace
    partial_result = (1 - entropy / num_qubits).real * 2 

    return partial_result

def calculate_entangling(data, sample=1024):
    """
        Returns the meyer-wallach entanglement measure for the given quantum states. 
        data: quantum states [bsz, [2] ** num_qubits]
    """
    # convert to density, then convert to statevector
    reshape_data = data.reshape(data.shape[0], -1)
    partial_results = [] # density convert to statevector，then calculate partial trace
    for i in range(reshape_data.shape[0]):
        state = Statevector(reshape_data[i,:])
        partial_result = _cal_partial_trace(state)
        partial_results.append(partial_result)

    average_entangling = np.mean(np.array(partial_results))
    
    return average_entangling

""" 
    numpy parallel: Entangling Capability 
"""
def _cal_partial_trace_p1(states):
    """ 
        Calculate partial trace of a quantum state.
    """
    num_qubits = len(states[0].dims())
    qb = list(range(num_qubits))
    qubits_list = list(itertools.combinations(qb, num_qubits-1))
    # trace_results = np.empty((len(qubits_list), states.shape[0]), dtype=complex)
    trace_results = np.empty(len(states), dtype=float)
    for i, state in enumerate(states):
        entropy = 0.0
        for num, qubits_combination in enumerate(qubits_list):
            dens = quantum_info.partial_trace(state, qubits_combination).data
            trace = np.trace(dens ** 2)
            entropy += trace

        trace_results[i] = (1 - entropy / num_qubits).real * 2
    average_entangling = np.mean(np.array(trace_results))

    return average_entangling

def calculate_entangling_p1(data, sample=1024):
    reshape_data = data.reshape(data.shape[0], -1)
    states = [Statevector(sample) for sample in reshape_data]
    average_entangling = _cal_partial_trace_p1(states)
    
    return average_entangling

def _cal_partial_trace_p2(state, qubits_list, num_qubits):
    
    """ 
        Calculate partial trace of a quantum state.
    """
    entropy = 0.0
    for qubits_combination in qubits_list:
        dens = quantum_info.partial_trace(state, qubits_combination).data
        trace = np.trace(dens ** 2, axis1=-1, axis2=-2)
        entropy += trace
    trace_result = (1 - entropy / num_qubits).real * 2
    return trace_result
    
def calculate_entangling_p2(data, sample=1024):
    reshape_data = data.reshape(data.shape[0], -1)
    states = [Statevector(sample) for sample in reshape_data]
    num_qubits = len(states[0].dims())
    qb = list(range(num_qubits))
    qubits_list = list(itertools.combinations(qb, num_qubits-1))
    trace_results = Parallel(n_jobs=-1)(delayed(_cal_partial_trace_p2)(state, qubits_list, num_qubits) for state in states)
    average_entangling = np.mean(np.array(trace_results))
    
    return average_entangling

