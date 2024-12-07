import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.plugins.qiskit_plugin import tq2qiskit
from models.classify_circ import model5, model1
from models.circ_layers import CascadeLayer
from models.classify_circ import classify_circ_dict

class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.jump = 1
            self.wire_reverse = True
            self.ops_all = tq.QuantumModuleList()
            for k in range(3):
                self.ops_all.append(tq.CRX(has_params=True,
                                   trainable=True))

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
            print(len(self.ops_all))

            # self.random_layer(q_device)
            # self.crx(q_device)
            for k in range(3):
                wires = [k, (k + self.jump) % self.n_wires]
                if self.wire_reverse:
                    wires.reverse()
                self.ops_all[k](q_device, wires=wires)
            

    def __init__(self, arch):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        # self.q_layer = self.QLayer()
        self.q_layer = model5(arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        self.q_device.reset_states(x.shape[0])
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        devi = x.device

        self.encoder(self.q_device, x)
        self.q_layer(self.q_device)
        x = self.measure(self.q_device)

        x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x

# format int to str
def format_int_to_str(num, length): 
    num_str = str(num)
    if len(num_str) < length:
        num_str = '0' * (length - len(num_str)) + num_str
    return num_str


def save_circuit_figure(model, arch):
    figure_path = './quantum_encoding/datasets/circuits_figure/'
    for num in range(1, 20, 1):
        arch['classify_circ'] = 'model' + str(num)
        model.q_layer = classify_circ_dict[arch['classify_circ']](arch=arch)
        circ = tq2qiskit(model.q_device, model.q_layer)
        figure = circ.draw(reverse_bits=True, output='mpl', style={'backgroundcolor': '#EEEEEE'})
        figure.suptitle('circuit_' + str(num), fontsize=20)
        figure.tight_layout()
        figure.savefig(figure_path + 'circuit_' + format_int_to_str(num, 2) + '.png')
