import torch
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf
# from torchquantum.plugins import (tq2qiskit_measurement,
#     qiskit_assemble_circs,
#     op_history2qiskit,
#     op_history2qiskit_expand_params,)
from models.classify_circ import classify_circ_dict
from models.enc_circ import encoder_circ_dict, encoder_op_list_name_dict

class TempleteCirc(tq.QuantumModule):
    """ stack multiple, encoder & classification network models """
    def __init__(self, arch):
        super().__init__()
        self.n_wires = arch['n_wires']
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        
        if arch['enc_layer'] == "general":
            if arch['enc_blocks'] == 1:
                arch['encode_method']='4x4_ryzxy' 
            elif arch['enc_blocks'] == 2:
                arch['encode_method'] = '4x8_ryzxy'

            arch['func_list'] = encoder_op_list_name_dict[arch['encode_method']]
            self.encoder = encoder_circ_dict[arch['enc_layer']](arch)

        elif arch['enc_layer'] == "phase":
            arch['func'] = tqf.rx
            self.encoder = encoder_circ_dict[arch['enc_layer']](arch)
        elif arch['enc_layer'] == "multiphase":
            arch['funcs'] = (['rx'] * 4 + ['ry'] * 4 + \
                             ['rz'] * 4 + ['rx'] * 4) * arch['enc_blocks']
            self.encoder = encoder_circ_dict[arch['enc_layer']](arch)


        elif arch['enc_layer'] in ["state_new", "state", "amplitude", "IQP"]:
            self.encoder = encoder_circ_dict[arch['enc_layer']](arch)
            
        elif arch['enc_layer'] == 'multiphase_z':
            arch['funcs'] =  (['rz'] * 4 + ['rx'] * 4 + ['ry'] * 4 + \
                              ['rz'] * 4) * arch['enc_blocks']
            self.encoder = encoder_circ_dict[arch['enc_layer']](arch)
        elif arch['enc_layer'] == 'multiphase_x':
            arch['funcs'] =  (['rx'] * 4 + ['rz'] * 4 + ['rx'] * 4 + \
                              ['rx'] * 4) * arch['enc_blocks']
            self.encoder = encoder_circ_dict[arch['enc_layer']](arch)

        elif arch['enc_layer'] == 'IQP_X':
            arch['rotation_gate'] = 'rx'
            self.encoder = encoder_circ_dict[arch['enc_layer']](arch)

        elif arch['enc_layer'] == 'IQP_Y':
            arch['rotation_gate'] = 'ry'
            self.encoder = encoder_circ_dict[arch['enc_layer']](arch)
       
        self.q_layer = classify_circ_dict[arch['classify_circ']](arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)
    
    @tq.static_support
    def forward(self, x, arch, 
                use_qiskit=False, save_encoded=False, print_encoded=False, info=None):
        self.q_device.reset_states(x.shape[0])
        bsz = x.shape[0]
        # import pdb; pdb.set_trace()
        
        if arch['enc_blocks'] == 1 and arch['data_type'] == "image":
            x = F.avg_pool2d(x, kernel_size=5).view(bsz, -1)
        elif arch['enc_blocks'] == 2 and arch['data_type'] == "image":
            x = F.avg_pool2d(x, kernel_size=4).view(bsz, -1)[:, :16*arch['enc_blocks']]
        elif arch['enc_blocks'] == 3 and arch['data_type'] == "image":
            x = F.avg_pool2d(x, kernel_size=4, stride=3).view(bsz, -1)[:, :16*arch['enc_blocks']]
        elif arch['data_type'] == "vowel":
            x = F.pad(x, (0, 16-x.shape[1]), mode='constant', value=0)

        qdev = tq.QuantumDevice(n_wires=self.n_wires,\
                                bsz=bsz, device=x.device, record_op=True)

        use_qiskit = arch['use_qiskit']
        pre_enc = x 
        if use_qiskit:
            # use qiskit to process the circuit
            # create the qiskit circuit for encoder
            self.encoder(qdev, x)
            op_history_parameterized = qdev.op_history
            encoded_states = qdev.states 
            qdev.reset_op_history()
            # import pdb; pdb.set_trace()
            encoder_circs = op_history2qiskit_expand_params(self.n_wires, op_history_parameterized, bsz=bsz)

            # create the qiskit circuit for q_layer
            self.q_layer(qdev)
            op_history_fixed = qdev.op_history
            qdev.reset_op_history()
            q_layer_circ = op_history2qiskit(self.n_wires, op_history_fixed)

            # create the qiskit circuit for measure
            measurement_circ = tq2qiskit_measurement(qdev, self.measure)
            measure_states = qdev.states 

            # assemble the encoder*bsz, trainble quantum layers, and measurement circuits
            assembeld_circs = qiskit_assemble_circs_new(
                encoder_circs, q_layer_circ, measurement_circ)
            
            # call the qiskit processor to process the circuit
            x0 = self.qiskit_processor.process_ready_circs(qdev, \
                                                           assembeld_circs).to(x.device)
            x = x0
        else:
            pre_enc = x 
            if ('train_empty' not in arch.keys()) or (arch['train_empty']==False):
                x = self.encoder(self.q_device, x)
            encoded_states = self.q_device.states
            self.q_layer(self.q_device)
            measure_states = self.q_device.states
            x = self.measure(self.q_device)

        if (arch['class_num'] == 2):
            x_pre_log = x.reshape(bsz, 2, x.shape[1]//2).sum(-1).squeeze()
        elif arch['class_num'] == 4:
            x_pre_log = x.squeeze()
        # x_log = F.log_softmax(x_pre_log, dim=1)
        x_log = F.softmax(x_pre_log, dim=1)

        return x_log, pre_enc, encoded_states, [x_pre_log, measure_states] 
