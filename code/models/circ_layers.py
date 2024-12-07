import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np


from typing import Iterable
from torchquantum.plugins.qiskit_macros import QISKIT_INCOMPATIBLE_FUNC_NAMES

__all__ = []

class CascadeLayer(tq.QuantumModule):
    '''pattern
    [0, 1], [0, 2], [0, 3]
    reverse:
    [3, 0], [2, 0], [1, 0]
    '''
    def __init__(self, op, n_wires: int, has_params=False, trainable=False,
                 wire_reverse=False, jump=1, circular=False):
        super().__init__()
        self.n_wires = n_wires
        self.op = op
        self.jump = jump
        self.wire_reverse = wire_reverse
        self.ops_all = tq.QuantumModuleList()
        if circular:
            n_ops = n_wires
        else:
            n_ops = n_wires - jump
        for k in range(n_ops):
            self.ops_all.append(op(has_params=has_params, 
                                   trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        # self.q_device = q_device

        if self.wire_reverse:
            for k in range(-1,len(self.ops_all)-1, 1):
                wires = [3, (k + self.jump) % self.n_wires]
                self.ops_all[k](q_device, wires=wires)
        else:
            for k in range(len(self.ops_all)):
                wires = [0, (k + self.jump) % self.n_wires]
                self.ops_all[k](q_device, wires=wires)


class CRLayers(tq.QuantumModule):
    '''
        using to circuit5 & 6 middle layers
    '''
    def __init__(self, op, n_wires: int, has_params=False, trainable=False):
        super().__init__()
        self.op = op
        self.cr_layers = tq.QuantumModuleList()
        for k in range(6):
            self.cr_layers.append(op(has_params=has_params, trainable=trainable))
        

    @tq.static_support
    def forward(self, q_device):
        wires_list = [[1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [2, 3]]
        for k in range(len(wires_list)):
            self.cr_layers[k](q_device, wires=wires_list[k])


class CRLayers2(tq.QuantumModule):
    '''
        using to circuit7 & 8 middle layers
    '''
    def __init__(self, op, n_wires: int, has_params=False, trainable=False):
        super().__init__()
        self.op = op
        self.cr_layers = tq.QuantumModuleList()
        for k in range(6):
            self.cr_layers.append(op(has_params=has_params, trainable=trainable))
        
    @tq.static_support
    def forward(self, q_device):
        wires_list = [[1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [2, 3]]
        for k in range(len(wires_list)):
            self.cr_layers[k](q_device, wires=wires_list[k])


class ParallelCR(tq.QuantumModule):
    '''
        using to circuit7 & 8 middle layers
    '''
    def __init__(self, op, n_wires: int, has_params=False, trainable=False):
        super().__init__()
        self.op = op
        self.cr_layers = tq.QuantumModuleList()
        for k in range(2):
            self.cr_layers.append(op(has_params=has_params, trainable=trainable))
        
    @tq.static_support
    def forward(self, q_device):
        wires_list = [[0, 1], [2, 3]]
        for k in range(len(wires_list)):
            self.cr_layers[k](q_device, wires=wires_list[k])


class ParallelR(tq.QuantumModule):
    '''
        using to circuit7 & 8 middle layers
    '''
    def __init__(self, op, n_wires: int, has_params=False, trainable=False):
        super().__init__()
        self.op = op
        self.r_layers = tq.QuantumModuleList()
        for k in range(2):
            self.r_layers.append(op(has_params=has_params, trainable=trainable))
        
    @tq.static_support
    def forward(self, q_device):
        wires_list = [[1], [2]]
        for k in range(len(wires_list)):
            self.r_layers[k](q_device, wires=wires_list[k])

class CascadeCRLayer(tq.QuantumModule):
    '''
        for circuit 13~15 latter part CR layers
    '''
    def __init__(self, op, n_wires: int, has_params=False, trainable=False):
        super().__init__()
        self.op = op
        self.cr_layers = tq.QuantumModuleList()

        for k in range(4):
            self.cr_layers.append(op(has_params=True, trainable=True))
        
    @tq.static_support
    def forward(self, q_device):
        wires_list = [[0, 1], [3, 0], [2, 3], [1, 2]]
        for k in range(len(wires_list)):
            self.cr_layers[k](q_device, wires=wires_list[k])


""" IQP layers """
class Op1QAllLayer_R(tq.QuantumModule):
    def __init__(self, op, n_wires: int, has_params=False, trainable=False):
        super().__init__()
        self.n_wires = n_wires
        self.op = op
        self.ops_all = tq.QuantumModuleList()
        for k in range(n_wires):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))
    @tq.static_support
    def forward(self, q_device, params):
        for i in range(self.n_wires):
            self.ops_all[i].params[0] = params[:, i]
        for k in range(self.n_wires):
            self.ops_all[k](q_device, wires=k)


   

class Op2QAllLayer_R(tq.QuantumModule):
    """pattern:
    circular = False
    jump = 1: [0, 1], [1, 2], [2, 3], [3, 4], [4, 5]
    jump = 2: [0, 2], [1, 3], [2, 4], [3, 5]
    jump = 3: [0, 3], [1, 4], [2, 5]
    jump = 4: [0, 4], [1, 5]
    jump = 5: [0, 5]

    circular = True
    jump = 1: [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]
    jump = 2: [0, 2], [1, 3], [2, 4], [3, 5], [4, 0], [5, 1]
    jump = 3: [0, 3], [1, 4], [2, 5], [3, 0], [4, 1], [5, 2]
    jump = 4: [0, 4], [1, 5], [2, 0], [3, 1], [4, 2], [5, 3]
    jump = 5: [0, 5], [1, 0], [2, 1], [3, 2], [4, 3], [5, 4]
    """
    def __init__(self, op, n_wires: int, has_params=False, trainable=False,
                 wire_reverse=False, jump=1, circular=False):
        super().__init__()
        self.n_wires = n_wires
        self.jump = jump
        self.circular = circular
        self.op = op
        self.ops_all = tq.QuantumModuleList()

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse

        if circular:
            n_ops = n_wires
        else:
            n_ops = n_wires - jump
        for k in range(n_ops):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    @tq.static_support
    def forward(self, q_device, x):
        for k in range(len(self.ops_all)):
            wires = [k, (k + self.jump) % self.n_wires]
            if self.wire_reverse:
                wires.reverse()
            self.ops_all[k](q_device, wires=wires, params=x)
