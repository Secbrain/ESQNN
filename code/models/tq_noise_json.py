from qiskit import IBMQ
import torch
from torchquantum.utils import get_provider
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from torchquantum.noise_model import (apply_readout_error_func, 
                                      NoiseModelTQ, 
                                      NoiseModelTQReadoutOnly, 
                                      NoiseModelTQQErrorOnly)
from qiskit.providers.aer.noise import NoiseModel
from .json_noise_model import IBMBackend_NoiseModel


class NoiseModelTQJson(NoiseModelTQ):
    """
    add noise from json
    """
    def __init__(self,
                 noise_model_name,
                 n_epochs,
                 mean=0.,
                 std=1.,
                 noise_total_prob=None,
                 ignored_ops=('id', 'kraus', 'reset'),
                 prob_schedule=None,
                 prob_schedule_separator=None,
                 factor=None,
                 add_thermal=True,
                 json_file_path=None,
                 add_gate=True,
                 add_readout=True,
                 ):
        self.noise_model_name = noise_model_name
        provider = get_provider(backend_name=noise_model_name)
        backend = provider.get_backend(noise_model_name)
        self.noise_model = IBMBackend_NoiseModel.from_backend_json(json_file_path, backend, 
                                                                   gate_error=add_gate, 
                                                                   readout_error=add_readout, 
                                                                   thermal_relaxation=add_thermal)
        self.noise_model_dict = self.noise_model.to_dict()

        self.mean = mean
        self.std = std
        self.is_add_noise = True
        self.v_c_reg_mapping = None
        self.p_c_reg_mapping = None
        self.p_v_reg_mapping = None
        self.orig_noise_total_prob = noise_total_prob
        self.noise_total_prob = noise_total_prob
        self.mode = 'train'
        self.ignored_ops = ignored_ops

        self.parsed_dict = self.parse_noise_model_dict(self.noise_model_dict)
        self.parsed_dict = self.clean_parsed_noise_model_dict(
            self.parsed_dict, ignored_ops)
        self.n_epochs = n_epochs
        self.prob_schedule = prob_schedule
        self.prob_schedule_separator = prob_schedule_separator
        self.factor = factor
        self.add_gate = add_gate and self.is_add_noise
        self.add_readout = add_readout
        self.add_thermal = add_thermal

    def apply_readout_error(self, x):
        c2p_mapping = self.p_c_reg_mapping['c2p']
        measure_info = self.parsed_dict['measure']

        return apply_readout_error_func(x, c2p_mapping, measure_info)
    
    def add_phase_noise(self, phase):
        if self.mode == 'train' and self.add_gate:
            if self.factor is None:
                factor = 1
            else:
                factor = self.factor
            phase = phase + torch.randn(phase.shape, device=phase.device) * \
                self.std * factor + self.mean

        return phase
    

def make_noise_model_json_tq():
    print(f"make_noise_model_json_tq: {configs.qiskit.test_noise_category}")

    noise_model_tq = NoiseModelTQJson(configs.qiskit.noise_model_name, 
                                      configs.train.n_epochs, 
                                      noise_total_prob=getattr(configs.train, 
                                                               'noise_total_prob',
                                                               None),
                                      ignored_ops=configs.train.ignored_noise_ops,
                                      prob_schedule=getattr(configs.train, 
                                                            'noise_prob_schedule',
                                                            None),
                                      prob_schedule_separator=getattr(configs.train, 
                                                                      'noise_prob_schedule_separator', 
                                                                      None),
                                      factor=getattr(configs.train, 
                                                     'noise_factor', 
                                                     None),
                                      add_thermal=getattr(configs.train, 
                                                          'noise_add_thermal', 
                                                          True))

    return noise_model_tq
