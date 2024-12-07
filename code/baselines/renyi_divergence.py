import numpy as np

__all__ = ['mutual_info_renyi', 'average_encode_state', 'cal_renyi_divergence', 'compare_state_distance']

def mutual_info_renyi(q_x, c_y, alpha=2):
    """
    Calculate mutual information between quantum states and classical data.
    q_x: quantum states
    c_y: classical data
    """
    # Calculate joint probability distribution and joint entropy
    qx_probabilities = np.abs(q_x.reshape(-1))**2
    cy_probabilities, bin_edges = np.histogram(c_y, bins=np.arange(c_y.min(), c_y.max()+2), density=True)
    H_QC = -np.sum(np.kron(qx_probabilities, cy_probabilities) * np.log2(np.kron(qx_probabilities, cy_probabilities)))

    # Calculate the Renyi entropy of the quantum state
    temp_matrix = q_x.reshape(-1, 2, 2)
    density_matrix = temp_matrix[0]
    for i in range(1, temp_matrix.shape[0]):
        density_matrix = np.kron(density_matrix, temp_matrix[i])
    H_rho = (1 / (1 - alpha)) * np.log2(np.trace(np.power(density_matrix.real, alpha)))

    # Calculate the Renyi entropy of the classical variable
    H_X = (1 / (1 - alpha)) * np.log2(np.sum(np.power(cy_probabilities, alpha)))
    
    # Calculate mutual information
    I_rho_X = H_rho + H_X - H_QC
    
    return I_rho_X

def average_encode_state(train_x):
    d1, d2 = train_x.shape
    density_matrices = np.reshape(train_x, [d1, d2, 1]) @ np.reshape(np.conj(train_x), [d1, 1, d2])
    return np.mean(density_matrices, axis=0)

def cal_renyi_divergence(states):
    '''
    Calculate the Renyi divergence of a batch of quantum states.
    states: [batch_size, [2] ** num_qubits]
    '''
    reshape_states = np.reshape(states, [states.shape[0], -1])
    bsz, num_qubits = reshape_states.shape
    average_state = average_encode_state(reshape_states.real)
    Q_2Renyi = np.log2(np.trace(average_state @ average_state) * 2 ** num_qubits) # scalar

    return Q_2Renyi

def compare_state_distance(states):
    """
    Calculate the average distance between states and the max distance.
    """
    num_states = states.shape[0]
    avg_dist = []
    max_dist = 0
    for i in range(num_states):
        for j in range(i+1, num_states):
            state_distance = np.linalg.norm(states[i].reshape(-1) - states[j].reshape(-1))
            avg_dist.append(state_distance)
            if state_distance > max_dist:
                max_dist = state_distance
    return max_dist, avg_dist