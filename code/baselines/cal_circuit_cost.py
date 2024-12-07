''' Calculate Circuit cost'''
from qiskit import IBMQ
import torchquantum as tq
from torchquantum.plugins import (
    tq2qiskit_expand_params,
    tq2qiskit,
    qiskit2tq,
    tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,
)
from torchquantum.plugins import QiskitProcessor
from models_custom.quan_models import model6, model11, model14
IBMQ.load_account()


arch = {'n_wires': 4, 'cls_blocks': 2,
        'backend_list': ['ibmq_quito', 'ibmq_lima', 'ibmq_belem', 'ibmq_manila'],
        'models': ['model6', 'model11', 'model14'],}

for model in arch['models']:
    arch['q_layer'] = model
    if arch['q_layer'] == 'model6':
        q_layer = model6(arch)
    elif arch['q_layer'] == 'model11':
        q_layer = model11(arch)
    elif arch['q_layer'] == 'model14':
        q_layer = model14(arch)
    qdev = tq.QuantumDevice(n_wires=arch['n_wires'])
    circ = tq2qiskit(qdev, q_layer)
    circ.measure_all()
    circ_ops, circ_depth, circ_size = circ.count_ops(), circ.depth(), circ.size()
    print('origin circuit: ',  circ_ops, circ_depth, circ_size)
    with open('./circuit_cost/circuit_execution.txt', 'a+') as file:
        file.write(f'{model}\n'+f'count_ops={circ_ops}, circ_depth={circ_depth}, total_gates_num={circ_size}\n' )
    for backend_name in arch['backend_list']:
        processor = QiskitProcessor(use_real_qc=True, backend_name=backend_name)
        circ_transpiled = processor.transpile(circs=circ)
        circ_trans_ops = circ_transpiled.count_ops()
        circ_trans_depth = circ_transpiled.depth()
        circ_trans_size = circ_transpiled.size()
        print(backend_name, 'transpiled circuit: ', circ_trans_ops, circ_trans_depth, circ_trans_size)
        with open('./circuit_cost/circuit_execution.txt', 'a+') as file:
            file.write(f'{backend_name}\n'+\
                       f'count_ops={circ_trans_ops}, circ_depth={circ_trans_depth}, total_gates_num={circ_trans_size}\n' )