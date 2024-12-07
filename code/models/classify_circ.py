import torchquantum as tq
import torch.nn.functional as F
from torchpack.utils.logging import logger

from models.circ_layers import CascadeLayer, CRLayers, ParallelCR, ParallelR, CascadeCRLayer

class model1(tq.QuantumModule):
    '''
        rx, rz: total 2 layers
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.rx_layers = tq.QuantumModuleList()
        self.rz_layers = tq.QuantumModuleList()
    
        for k in range(arch['cls_blocks']):
            self.rx_layers.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                  has_params=True, trainable=True))
            self.rz_layers.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                  has_params=True, trainable=True))
        
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            # self.rx_layers[k](self.q_device)
            # self.rz_layers[k](self.q_device)
            self.rx_layers[k](q_device)
            self.rz_layers[k](q_device)

class model2(tq.QuantumModule):
    '''
        rx, rz, cnot
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.rx_layers = tq.QuantumModuleList()
        self.rz_layers = tq.QuantumModuleList()
        self.cx_layers = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.rx_layers.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires, 
                                has_params=True, trainable=True))
            self.cx_layers.append(
                tq.Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires,
                                has_params=True, trainable=True))
    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.rx_layers[k](q_device)
            self.rz_layers[k](q_device)
            self.cx_layers[k](q_device)
            
class model3(tq.QuantumModule):
    '''
        rx, rz, crz
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.rx_layers = tq.QuantumModuleList()
        self.rz_layers = tq.QuantumModuleList()
        self.crz_layers = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.rx_layers.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crz_layers.append(
                tq.Op2QAllLayer(op=tq.CRZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.rx_layers[k](q_device)
            self.rz_layers[k](q_device)
            self.crz_layers[k](q_device)

class model4(tq.QuantumModule):
    '''
        rx, rz, crx
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.rx_layers = tq.QuantumModuleList()
        self.rz_layers = tq.QuantumModuleList()
        self.crx_layers = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.rx_layers.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))  
            self.rz_layers.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crx_layers.append(
                tq.Op2QAllLayer(op=tq.CRX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.rx_layers[k](q_device)
            self.rz_layers[k](q_device)
            self.crx_layers[k](q_device)

class model5(tq.QuantumModule):
    '''
        rx, rz, crz
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.rx_layers0 = tq.QuantumModuleList()
        self.rz_layers0 = tq.QuantumModuleList()
        self.crz_layers = tq.QuantumModuleList() # cascade layer
        self.middle_layers = tq.QuantumModuleList()
        self.crz_reverse_layers = tq.QuantumModuleList()
        self.rx_layers1 = tq.QuantumModuleList()
        self.rz_layers1 = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.rx_layers0.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers0.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crz_layers.append(
                CascadeLayer(op=tq.CRZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.middle_layers.append(
                CRLayers(op=tq.CRZ, n_wires=self.n_wires,
                         has_params=True, trainable=True))
            self.crz_reverse_layers.append(
                CascadeLayer(op=tq.CRZ, n_wires=self.n_wires,
                             has_params=True, trainable=True, wire_reverse=True))
            self.rx_layers1.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers1.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            
    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.rx_layers0[k](q_device)
            self.rz_layers0[k](q_device)
            self.crz_layers[k](q_device)
            self.middle_layers[k](q_device)
            self.crz_reverse_layers[k](q_device)
            self.rx_layers1[k](q_device)
            self.rz_layers1[k](q_device)

class model6(tq.QuantumModule):
    '''
        rx, rz, crz
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.rx_layers0 = tq.QuantumModuleList()
        self.rz_layers0 = tq.QuantumModuleList()
        self.crx_layers = tq.QuantumModuleList() # cascade layer
        self.middle_layers = tq.QuantumModuleList()
        self.crx_reverse_layers = tq.QuantumModuleList()
        self.rx_layers1 = tq.QuantumModuleList()
        self.rz_layers1 = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.rx_layers0.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers0.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crx_layers.append(
                CascadeLayer(op=tq.CRX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.middle_layers.append(
                CRLayers(op=tq.CRX, n_wires=self.n_wires,
                         has_params=True, trainable=True))
            self.crx_reverse_layers.append(
                CascadeLayer(op=tq.CRX, n_wires=self.n_wires,
                             has_params=True, trainable=True, wire_reverse=True))
            self.rx_layers1.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers1.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            
    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.rx_layers0[k](q_device)
            self.rz_layers0[k](q_device)
            self.crx_layers[k](q_device)
            self.middle_layers[k](q_device)
            self.crx_reverse_layers[k](q_device)
            self.rx_layers1[k](q_device)
            self.rz_layers1[k](q_device)

class model7(tq.QuantumModule):
    '''
        rx, rz, crz, rx, rz, crx
    '''
    
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.rx_layers0 = tq.QuantumModuleList()
        self.rz_layers0 = tq.QuantumModuleList()
        self.crz_layers0 = tq.QuantumModuleList()
        self.rx_layers1 = tq.QuantumModuleList()
        self.rz_layers1 = tq.QuantumModuleList()
        self.crz_layers1 = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.rx_layers0.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers0.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crz_layers0.append(ParallelCR(op=tq.CRZ, n_wires=self.n_wires,
                                                has_params=True, trainable=True))
            self.rx_layers1.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers1.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crz_layers1.append(tq.CRZ(has_params=True, trainable=True))

    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.rx_layers0[k](q_device)
            self.rz_layers0[k](q_device)
            self.crz_layers0[k](q_device)
            # self.crz_layers0[k](q_device, wires=[2, 3])
            self.rx_layers1[k](q_device)
            self.rz_layers1[k](q_device)
            self.crz_layers1[k](q_device, wires=[1, 2])

class model8(tq.QuantumModule):
    '''
        rx, rz, crz, rx, rz, crx
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.rx_layers0 = tq.QuantumModuleList()
        self.rz_layers0 = tq.QuantumModuleList()
        self.crx_layers0 = tq.QuantumModuleList()
        self.rx_layers1 = tq.QuantumModuleList()
        self.rz_layers1 = tq.QuantumModuleList()
        self.crx_layers1 = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.rx_layers0.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers0.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crx_layers0.append(ParallelCR(op=tq.CRX, n_wires=self.n_wires,
                                                has_params=True, trainable=True))
            self.rx_layers1.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers1.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crx_layers1.append(tq.CRX(has_params=True, trainable=True))

    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.rx_layers0[k](q_device)
            self.rz_layers0[k](q_device)
            self.crx_layers0[k](q_device)
            self.rx_layers1[k](q_device)
            self.rz_layers1[k](q_device)
            self.crx_layers1[k](q_device, wires=[1, 2])

class model9(tq.QuantumModule):
    '''
        h, cz, rx
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.h_layers = tq.QuantumModuleList()
        self.cz_layers = tq.QuantumModuleList()
        self.rx_layers = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.h_layers.append(
                tq.Op1QAllLayer(op=tq.Hadamard, n_wires=self.n_wires,
                                has_params=False, trainable=False))
            self.cz_layers.append(
                tq.Op2QAllLayer(op=tq.CZ, n_wires=self.n_wires,
                                has_params=False, trainable=False))
            self.rx_layers.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))

    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.h_layers[k](q_device)
            self.cz_layers[k](q_device)
            self.rx_layers[k](q_device)

class model10(tq.QuantumModule):
    '''
        ry, cz, ry
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.ry_layers0 = tq.QuantumModuleList()
        self.cz_layers = tq.QuantumModuleList()
        self.cz_layers1 = tq.QuantumModuleList()
        self.ry_layers1 = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.ry_layers0.append(
                tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.cz_layers.append(
                tq.Op2QAllLayer(op=tq.CZ, n_wires=self.n_wires,
                                has_params=False, trainable=False))
            self.cz_layers1.append(tq.CZ(has_params=False, trainable=False))
            self.ry_layers1.append(
                tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                has_params=True, trainable=True))
    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.ry_layers0[k](q_device)
            self.cz_layers[k](q_device)
            self.cz_layers1[k](q_device, wires=[3, 0])
            self.ry_layers1[k](q_device)

class model11(tq.QuantumModule):
    '''
        ry0, rz0, cnot0, ry1, rz1, cnot1
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.ry_layers0 = tq.QuantumModuleList()
        self.rz_layers0 = tq.QuantumModuleList()
        self.cnot_layers0 = tq.QuantumModuleList()
        self.ry_layers1 = tq.QuantumModuleList()
        self.rz_layers1 = tq.QuantumModuleList()
        self.cnot_layers1 = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.ry_layers0.append(
                tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers0.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            
            self.cnot_layers0.append(ParallelCR(op=tq.CNOT, n_wires=self.n_wires,
                                                has_params=True, trainable=True))
            
            self.ry_layers1.append(ParallelR(op=tq.RY, n_wires=self.n_wires,
                                                has_params=True, trainable=True))

            self.rz_layers1.append(ParallelR(op=tq.RZ, n_wires=self.n_wires,
                                                has_params=True, trainable=True))

            self.cnot_layers1.append(tq.CNOT(has_params=False, trainable=False))

        
    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.ry_layers0[k](q_device)
            self.rz_layers0[k](q_device)
            self.cnot_layers0[k](q_device)
            self.ry_layers1[k](q_device)
            self.rz_layers1[k](q_device)
            self.cnot_layers1[k](q_device, wires=[1, 2])            

class model12(tq.QuantumModule):
    '''
        ry0, rz0, cz0, ry1, rz1, cz1
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.ry_layers0 = tq.QuantumModuleList()
        self.rz_layers0 = tq.QuantumModuleList()
        self.cz_layers0 = tq.QuantumModuleList()
        self.ry_layers1 = tq.QuantumModuleList()
        self.rz_layers1 = tq.QuantumModuleList()
        self.cz_layers1 = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.ry_layers0.append(
                tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers0.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            
            self.cz_layers0.append(ParallelCR(op=tq.CZ, n_wires=self.n_wires,
                                                has_params=True, trainable=True))
            
            self.ry_layers1.append(ParallelR(op=tq.RY, n_wires=self.n_wires,
                                                has_params=True, trainable=True))

            self.rz_layers1.append(ParallelR(op=tq.RZ, n_wires=self.n_wires,
                                                has_params=True, trainable=True))

            self.cz_layers1.append(tq.CZ(has_params=False, trainable=False))

        
    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.ry_layers0[k](q_device)
            self.rz_layers0[k](q_device)
            self.cz_layers0[k](q_device)
            self.ry_layers1[k](q_device)
            self.rz_layers1[k](q_device)
            self.cz_layers1[k](q_device, wires=[1, 2])

class model13(tq.QuantumModule):
    '''
        ry0, crz0, ry1, crz1
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.ry_layers0 = tq.QuantumModuleList()
        self.crz_layers0 = tq.QuantumModuleList()
        self.crz_layers0_1 = tq.QuantumModuleList()
        self.ry_layers1 = tq.QuantumModuleList()
        self.crz_layers1 = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.ry_layers0.append(
                tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crz_layers0.append(tq.CRZ(has_params=True, trainable=True))
            self.crz_layers0_1.append(
                tq.Op2QAllLayer(op=tq.CRZ, n_wires=self.n_wires,
                                has_params=True, trainable=True, wire_reverse=True))
            self.ry_layers1.append(
                tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crz_layers1.append(CascadeCRLayer(op=tq.CRZ, n_wires=self.n_wires,
                                                has_params=True, trainable=True))
    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.ry_layers0[k](q_device)
            self.crz_layers0[k](q_device, wires=[0, 3])
            self.crz_layers0_1[k](q_device)
            self.ry_layers1[k](q_device)
            self.crz_layers1[k](q_device)

class model14(tq.QuantumModule):
    '''
        ry0, crx0, ry1, crx1
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.ry_layers0 = tq.QuantumModuleList()
        self.crx_layers0 = tq.QuantumModuleList()
        self.crx_layers0_1 = tq.QuantumModuleList()
        self.ry_layers1 = tq.QuantumModuleList()
        self.crx_layers1 = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.ry_layers0.append(
                tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crx_layers0.append(tq.CRX(has_params=True, trainable=True))
            self.crx_layers0_1.append(
                tq.Op2QAllLayer(op=tq.CRX, n_wires=self.n_wires,
                                has_params=True, trainable=True, wire_reverse=True))
            self.ry_layers1.append(
                tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crx_layers1.append(CascadeCRLayer(op=tq.CRX, n_wires=self.n_wires,
                                                has_params=True, trainable=True))
    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.ry_layers0[k](q_device)
            self.crx_layers0[k](q_device, wires=[0, 3])
            self.crx_layers0_1[k](q_device)
            self.ry_layers1[k](q_device)
            self.crx_layers1[k](q_device)

class model15(tq.QuantumModule):
    '''
        ry0, cnot0, ry1, cnot1
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.ry_layers0 = tq.QuantumModuleList()
        self.cnot_layers0 = tq.QuantumModuleList()
        self.cnot_layers0_1 = tq.QuantumModuleList()
        self.ry_layers1 = tq.QuantumModuleList()
        self.cnot_layers1 = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.ry_layers0.append(
                tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.cnot_layers0.append(tq.CNOT(has_params=True, trainable=True))
            self.cnot_layers0_1.append(
                tq.Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires,
                                has_params=True, trainable=True, wire_reverse=True))
            self.ry_layers1.append(
                tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.cnot_layers1.append(CascadeCRLayer(op=tq.CNOT, n_wires=self.n_wires,
                                                has_params=True, trainable=True))
    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.ry_layers0[k](q_device)
            self.cnot_layers0[k](q_device, wires=[0, 3])
            self.cnot_layers0_1[k](q_device)
            self.ry_layers1[k](q_device)
            self.cnot_layers1[k](q_device)

class model16(tq.QuantumModule):
    '''
        rx, rz0, rz1, rz2 
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.rx_layers = tq.QuantumModuleList()
        self.rz_layers0 = tq.QuantumModuleList()
        self.crz_layers = tq.QuantumModuleList()
        self.crz_layers1 = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.rx_layers.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers0.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crz_layers.append(ParallelCR(op=tq.CRZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crz_layers1.append(tq.CRZ(has_params=True, trainable=True))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.rx_layers[k](q_device)
            self.rz_layers0[k](q_device)
            self.crz_layers[k](q_device)
            self.crz_layers1[k](q_device, wires=[1, 2])

class model17(tq.QuantumModule):
    '''
        rx, rz0, rz1, rz2 
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.rx_layers = tq.QuantumModuleList()
        self.rz_layers0 = tq.QuantumModuleList()
        self.crx_layers = tq.QuantumModuleList()
        self.crx_layers1 = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.rx_layers.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers0.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crx_layers.append(ParallelCR(op=tq.CRX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crx_layers1.append(tq.CRX(has_params=True, trainable=True))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.rx_layers[k](q_device)
            self.rz_layers0[k](q_device)
            self.crx_layers[k](q_device)
            self.crx_layers1[k](q_device, wires=[1, 2])

class model18(tq.QuantumModule):
    '''
        rx, rz, crz
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.rx_layers = tq.QuantumModuleList()
        self.rz_layers = tq.QuantumModuleList()
        self.crz_layers0 = tq.QuantumModuleList()
        self.crz_layers1 = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.rx_layers.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crz_layers0.append(tq.CRZ(has_params=True, trainable=True))
            self.crz_layers1.append(
                tq.Op2QAllLayer(op=tq.CRZ, n_wires=self.n_wires,
                                has_params=True, trainable=True, wire_reverse=True))
        
    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.rx_layers[k](q_device)
            self.rz_layers[k](q_device)
            self.crz_layers0[k](q_device, wires=[0, 3])
            self.crz_layers1[k](q_device)

class model19(tq.QuantumModule):
    '''
        rx, rz, crx
    '''
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.rx_layers = tq.QuantumModuleList()
        self.rz_layers = tq.QuantumModuleList()
        self.crx_layers0 = tq.QuantumModuleList()
        self.crx_layers1 = tq.QuantumModuleList()

        for k in range(arch['cls_blocks']):
            self.rx_layers.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.rz_layers.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.crx_layers0.append(tq.CRX(has_params=True, trainable=True))
            self.crx_layers1.append(
                tq.Op2QAllLayer(op=tq.CRX, n_wires=self.n_wires,
                                has_params=True, trainable=True, wire_reverse=True))
        
    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        for k in range(self.arch['cls_blocks']):
            self.rx_layers[k](q_device)
            self.rz_layers[k](q_device)
            self.crx_layers0[k](q_device, wires=[0, 3])
            self.crx_layers1[k](q_device)

classify_circ_dict = {
    'model1': model1,
    'model2': model2,
    'model3': model3,
    'model4': model4,
    'model5': model5,
    'model6': model6,
    'model7': model7,
    'model8': model8,
    'model9': model9,
    'model10': model10,
    'model11': model11,
    'model12': model12,
    'model13': model13,
    'model14': model14,
    'model15': model15,
    'model16': model16,
    'model17': model17,
    'model18': model18,    
    'model19': model19
}