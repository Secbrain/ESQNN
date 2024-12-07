import torch
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.plugins import (tq2qiskit_expand_params,
                                  tq2qiskit,
                                  tq2qiskit_measurement,
                                  qiskit_assemble_circs)
from .classify_circ import classify_circ_dict
from .tq_noise_json import make_noise_model_json_tq


class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires=4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50,
                                               wires=list(range(self.n_wires)))

            # gates with trainable parameters
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            """
            1. To convert tq QuantumModule to qiskit or run in the static
            model, need to:
                (1) add @tq.static_support before the forward
                (2) make sure to add
                    static=self.static_mode and
                    parent_graph=self.graph
                    to all the tqf functions, such as tqf.hadamard below
            """
            self.q_device = q_device

            self.random_layer(self.q_device)

            for i in range(self.n_wires//4):
                # some trainable gates (instantiated ahead of time)
                self.rx0(self.q_device, wires=4*i+0)
                self.ry0(self.q_device, wires=4*i+1)
                self.rz0(self.q_device, wires=4*i+3)
                self.crx0(self.q_device, wires=[4*i+0, 4*i+2])

                # add some more non-parameterized gates (add on-the-fly)
                tqf.hadamard(self.q_device, wires=4*i+3, static=self.static_mode,
                            parent_graph=self.graph)
                tqf.sx(self.q_device, wires=4*i+2, static=self.static_mode,
                    parent_graph=self.graph)
                tqf.cnot(self.q_device, wires=[4*i+3, 4*i+0], static=self.static_mode,
                        parent_graph=self.graph)

    def __init__(self, enc_layer, encode_method='4x4_ryzxy', n_wires=4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        if enc_layer == "general":
            self.encoder = tq.GeneralEncoder(
                tq.encoder_op_list_name_dict[encode_method])
        elif enc_layer == "phase":
            self.encoder = tq.PhaseEncoder(func=tqf.rx) 
        elif enc_layer == "multiphase":
            self.encoder = tq.MultiPhaseEncoder(['rx'] * 4 + ['ry'] * 4 +
                                                ['rz'] * 4 + ['rx'] * 4)
        elif enc_layer == "state":
            self.encoder = tq.StateEncoder()

        self.q_layer = self.QLayer(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, data_type="image", dataset_name="mnist", 
                use_qiskit=False, save_encoded=False, print_encoded=False, info=None):
        self.q_device.reset_states(x.shape[0])
        bsz = x.shape[0]
        if data_type == "vowel":
            x = F.pad(x, (0, 16-x.shape[1]), mode='reflect')
        elif dataset_name == "mnist":
            x = F.avg_pool2d(x, 6).view(bsz, 16, x.shape[1]).mean(dim=2)
        else:
            x = F.max_pool2d(x, 6).view(bsz, 16, x.shape[1]).mean(dim=2)
        # print(x.shape)
        devi = x.device

        if use_qiskit:
            encoder_circs = tq2qiskit_expand_params(self.q_device, x,
                                                    self.encoder.func_list)
            q_layer_circ = tq2qiskit(self.q_device, self.q_layer)
            measurement_circ = tq2qiskit_measurement(self.q_device,
                                                     self.measure)
            assembled_circs = qiskit_assemble_circs(encoder_circs,
                                                    q_layer_circ,
                                                    measurement_circ)
            x0 = self.qiskit_processor.process_ready_circs(
                self.q_device, assembled_circs).to(devi)
            # x1 = self.qiskit_processor.process_parameterized(
            #     self.q_device, self.encoder, self.q_layer, self.measure, x)
            # print((x0-x1).max())
            x = x0

        else:
            pre_enc = x
            self.encoder(self.q_device, x)
            encoded_states = self.q_device.states
            if save_encoded:
                torch.save(encoded_states, info["filepath"] + 'encoded_states/' + 
                           info["dataset_name"] + '_' + info["encode_method"] + 
                           '_encoded_batch_' + str(info["batch_index"]) + '.pth')
            if print_encoded:
                print(encoded_states)
                print(encoded_states.shape)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 2, x.shape[1]//2).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x, pre_enc, encoded_states



class TempleteCirc(tq.QuantumModule):
    def __init__(self, arch, enc_layer, encode_method='4x4_ryzxy'):
        super().__init__()
        self.n_wires = arch['n_wires']
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        if enc_layer == "general":
            self.encoder = tq.GeneralEncoder(
                tq.encoder_op_list_name_dict[encode_method])
        elif enc_layer == "phase":
            self.encoder = tq.PhaseEncoder(func=tqf.rx)
            self.encoder = tq.MultiPhaseEncoder(['rx'] * 4 + ['ry'] * 4 +
                                                ['rz'] * 4 + ['rx'] * 4)
        elif enc_layer == "state":
            self.encoder = tq.StateEncoder()
        self.q_layer = classify_circ_dict[arch['classify_circ']](arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.noise_model = make_noise_model_json_tq()
        
    @tq.static_support
    def forward(self, x, data_type="image", dataset_name="mnist", 
                use_qiskit=False, save_encoded=False, print_encoded=False, info=None):
        self.q_device.reset_states(x.shape[0])
        bsz = x.shape[0]
        if data_type == "vowel":
            x = F.pad(x, (0, 16-x.shape[1]), mode='constant', value=0)
        elif dataset_name == "mnist":
            x = F.avg_pool2d(x, 6).view(bsz, 16, x.shape[1]).mean(dim=2)
        else:
            x = F.max_pool2d(x, 6).view(bsz, 16, x.shape[1]).mean(dim=2)
        device_qiskit = self.device

        if use_qiskit:
            pass
        else:
            pre_enc = x # 编码前的数据
            self.encoder(self.q_device, x)
            encoded_states = self.q_device.states
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x_pre_log = x.reshape(bsz, 2, x.shape[1]//2).sum(-1).squeeze()
        # x_log = F.log_softmax(x_pre_log, dim=1)
        x_log = F.softmax(x_pre_log, dim=1)

        return x_log, pre_enc, encoded_states, x_pre_log 


class EmptyEncoder(tq.QuantumModule):
    def __init__(self, arch):
        super().__init__()
        self.n_wires = arch['n_wires']
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = classify_circ_dict[arch['classify_circ']](arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)
    
    @tq.static_support
    def forward(self, x, data_type="image", dataset_name="mnist", 
                use_qiskit=False, save_encoded=False, print_encoded=False, info=None):
        self.q_device.reset_states(x.shape[0])
        bsz = x.shape[0]
        if data_type == 'vowel':
            x = F.pad(x, (0, 16-x.shape[1]), mode='constant', value=0)
        elif dataset_name == "mnist":
            x = F.avg_pool2d(x, 6).view(bsz, 16, x.shape[1]).mean(dim=2)
        else:
            x = F.max_pool2d(x, 6).view(bsz, 16, x.shape[1]).mean(dim=2)
        device_qiskit = self.device

        if use_qiskit:
            pass
        else:
            pre_enc = x
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)
        
        x_pre_log = x.reshape(bsz, 2, x.shape[1]//2).sum(-1).squeeze()
        x_log = F.softmax(x_pre_log, dim=1)

        return x_log, pre_enc, None, x_pre_log


class PreEncoder(tq.QuantumModule):
    """ load encoded states from file """
    def __init__(self, arch):
        super().__init__()
        self.n_wires = arch['n_wires']
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = classify_circ_dict[arch['classify_circ']](arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)
    
    @tq.static_support
    def forward(self, x, data_type="image", dataset_name="mnist", 
                use_qiskit=False, save_encoded=False, print_encoded=False, info=None):
        self.q_device.reset_states(x.shape[0])
        bsz = x.shape[0]
        if data_type == 'vowel':
            x = F.pad(x, (0, 16-x.shape[1]), mode='constant', value=0)
        elif dataset_name == "mnist":
            x = F.avg_pool2d(x, 6).view(bsz, 16, x.shape[1]).mean(dim=2)
        else:
            x = F.max_pool2d(x, 6).view(bsz, 16, x.shape[1]).mean(dim=2)
        device_qiskit = self.device

        if use_qiskit:
            pass
        else:
            pre_enc = x
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)
        
        x_pre_log = x.reshape(bsz, 2, x.shape[1]//2).sum(-1).squeeze()
        x_log = F.softmax(x_pre_log, dim=1)

        return x_log, pre_enc, None, x_pre_log