import json
from numpy import inf, exp, allclose

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.models import BackendProperties
from qiskit.providers.aer.noise.errors.readout_error import ReadoutError 
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.circuit import Instruction, Delay
from qiskit.providers.aer.noise.passes import RelaxationNoisePass
from qiskit.providers.aer.noise.device.models import _excited_population, _truncate_t2_value
# from qiskit.providers.aer.noise.device.models import basic_device_gate_errors
# from qiskit.providers.aer.noise.device.models import basic_device_readout_errors

class IBMBackend_NoiseModel(NoiseModel):
    """
        properties, configuration load from backend,
        calculate readout & gate & thermal_relaxation load from json
    """
    def __init__(self, json_path, basis_gates=None):
        super().__init__(basis_gates)
        self.json_path = json_path # history data of noise_model.to_dict()
    
    @classmethod
    def from_backend_json(cls, json_path, backend, gate_error=True,
                    readout_error=True, thermal_relaxation=True,
                    temperature=0, gate_lengths=None,
                    gate_length_units='ns', standard_gates=None,
                    warnings=True):
        # refer Qiskit NoiseModel.from_backend
        # backend_interface_version = getattr(backend, "version", None)
        with open(json_path, "r") as f:
            backend_dict = json.load(f)
        
        backend_interface_version = int(backend_dict["backend_version"][0])
        if backend_interface_version <= 1:
            properties = backend.properties()
            configuration = backend.configuration()
            basis_gates = configuration.basis_gates
            num_qubits = configuration.num_qubits
            dt = getattr(configuration, "dt", 0) # qubit drive channel timestep in nanoseconds; 'dtm': measurement time
            if not properties:
                raise Exception('Qiskit backend {} does not have a '
                                 'BackendProperties'.format(backend))
        elif isinstance(backend, BackendProperties):
            properties = backend
            basis_gates = set()
            for prop in properties.gates:
                basis_gates.add(prop.gate)
            basis_gates = list(basis_gates)
            num_qubits = len(properties.qubits)
            dt = 0 # disable delay noise if dt is unknown
        else:
            raise Exception("{} is not a Qiskit backend or BackendProperties".format(backend))
        
        """
            load error from backend_dict(json) 
        """
        noise_model = NoiseModel(basis_gates=basis_gates) # print: Ideal

        # Add single-qubit readout errors
        if readout_error:
            # for qubits, error in basic_device_readout_errors()
            for qubit, error in device_readout_error(backend_dict):
                noise_model.add_readout_error(error, qubit, warnings=warnings)

        # add gate error
        gate_errors = device_gate_error(backend_dict, gate_error=gate_error,
                               thermal_relaxation=thermal_relaxation,
                               gate_lengths=gate_lengths,
                               gate_length_units=gate_length_units,
                               temperature=temperature,
                               standard_gates=standard_gates,
                               warnings=warnings)

        for name, qubits, error in gate_errors:
                    noise_model.add_quantum_error(error, name, qubits, warnings=warnings)
        
        # thermal relaxation
        if thermal_relaxation:
            # Add delay errors via RelaxationNiose pass
            try:
                
                excited_state_populations = [
                    _excited_population(
                        # freq=properties.frequency(q), temperature=temperature
                        freq=_check_for_dict_item(backend_dict['qubits'][q], 'frequency')['value'], 
                        temperature=temperature
                    ) for q in range(num_qubits)]
            except BackendPropertyError:
                excited_state_populations = None
            try:
                t1s = [_check_for_dict_item(backend_dict['qubits'][q], 'T1')['value'] for q in range(num_qubits)]
                t2s = [_check_for_dict_item(backend_dict['qubits'][q], 'T2')['value'] for q in range(num_qubits)]
                delay_pass = RelaxationNoisePass(
                    t1s=t1s,
                    t2s=[_truncate_t2_value(t1, t2) for t1, t2 in zip(t1s, t2s)],
                    dt=dt,
                    op_types=Delay,
                    excited_state_populations=excited_state_populations
                )
                noise_model._custom_noise_passes.append(delay_pass)
            except BackendPropertyError:
                # Device does not have the required T1 or T2 information
                # in its properties
                pass
        return noise_model
    




def _check_for_dict_item(qubit_props, name):
    filter_data = [item for item in qubit_props if item['name'] == name]
    if not filter_data:
        return None
    else:
        return filter_data[0]


# Add readout error
def readout_error_values(backend_dict):
    values = []
    for qubit_props in backend_dict["qubits"]:
        # readout_error
        params_readout_error = _check_for_dict_item(qubit_props, "readout_error")
        params_m1p0 = _check_for_dict_item(qubit_props, 'prob_meas1_prep0')
        params_m0p1 = _check_for_dict_item(qubit_props, 'prob_meas0_prep1')
        if ('value' in params_m1p0) and ('value' in params_m0p1):
            value = [params_m1p0['value'], params_m0p1['value']]
        elif 'value' in params_readout_error:
            value = [params_readout_error['value'], params_readout_error['value']]
        values.append(value)
    return values


def device_readout_error(backend_dict):
    errors = []
    for qubit, value in enumerate(readout_error_values(backend_dict)):
        if value is not None and not allclose(value, [0, 0]):
            properties = [[1 - value[0], value[0]], [value[1], 1 - value[1]]]
            errors.append(([qubit], ReadoutError(properties)))
    return errors


    
_NANOSECOND_UNITS = {'s': 1e9, 'ms': 1e6, 'µs': 1e3, 'us': 1e3, 'ns': 1}
_GHZ_UNITS = {'Hz': 1e-9, 'KHz': 1e-6, 'MHz': 1e-3, 'GHz': 1, 'THz': 1e3}
def thermal_relaxation_values_json(backend_dict):
    values = []
    for qubit_props in backend_dict['qubits']:
        # pylint: disable=invalid-name
        # Default values
        t1, t2, freq = inf, inf, inf

        # Get the readout error value
        t1_params = _check_for_dict_item(qubit_props, 'T1')
        t2_params = _check_for_dict_item(qubit_props, 'T2')
        freq_params = _check_for_dict_item(qubit_props, 'frequency')

        # Load values from parameters
        if 'value' in t1_params:
            t1 = t1_params['value']
            if 'unit' in t1_params:
                # Convert to nanoseconds
                t1 *= _NANOSECOND_UNITS.get(t1_params['unit'], 1)
        if 'value' in t2_params:
            t2 = t2_params['value']
            if 'unit' in t2_params:
                # Convert to nanoseconds
                t2 *= _NANOSECOND_UNITS.get(t2_params['unit'], 1)
        if 'value' in freq_params:
            freq = freq_params['value']
            if 'unit' in freq_params:
                # Convert to Gigahertz
                freq *= _GHZ_UNITS.get(freq_params['unit'], 1)

        values.append((t1, t2, freq))
    return values

def gate_param_values_json(backend_dict):
    values = []
    for gate in backend_dict['gates']:
        name = gate['gate']
        qubits = gate['qubits']
        # Check for gate time information
        gate_length = None  # default value
        time_param = _check_for_dict_item(gate['parameters'], 'gate_length')
        if 'value' in time_param:
            gate_length = time_param['value']
            if 'unit' in time_param:
                # Convert gate time to ns
                gate_length *= _NANOSECOND_UNITS.get(time_param['unit'], 1)
        # Check for gate error information
        gate_error = None  # default value
        error_param = _check_for_dict_item(gate['parameters'], 'gate_error')

        if error_param and 'value' in error_param:
            gate_error = error_param['value']
        values.append((name, qubits, gate_length, gate_error))
    return values

from qiskit.providers.aer.noise.device.models import _device_thermal_relaxation_error, _device_depolarizing_error
def device_gate_error(backend_dict,
                     gate_error=True,
                     thermal_relaxation=True,
                     gate_lengths=None,
                     gate_length_units='ns',
                     temperature=0,
                     standard_gates=None,
                     warnings=True):
    # initialize empty errors
    depol_error = None
    relax_error = None
    # Generate custom gate time dict
    custom_times = {}
    relax_params = []
    if thermal_relaxation:
         # If including thermal relaxation errors load
        # T1, T2, and frequency values from backend_dict
        relax_params = thermal_relaxation_values_json(backend_dict) # [(T1, T2, frequency) * num_qubits]
        # If we are specifying custom gate times include
        # them in the custom times dict
        if gate_lengths:
            for name, qubits, value in gate_lengths:
                # Convert all gate lengths to nanosecond units
                time = value * _NANOSECOND_UNITS[gate_length_units]
                if name in custom_times:
                    custom_times[name].append((qubits, time))
                else:
                    custom_times[name] = [(qubits, time)]
    # Get the device gate parameters from backend_dict
    device_gate_params = gate_param_values_json(backend_dict)

    # Construct quantum errors
    errors = []
    for name, qubits, gate_length, error_param in device_gate_params:
        # Check for custom gate time
        relax_time = gate_length
        # Override with custom value
        if name in custom_times:
            filtered = [
                val for q, val in custom_times[name]
                if q is None or q == qubits
            ]
            if filtered:
                # get first value
                relax_time = filtered[0]
        # Get relaxation error
        if thermal_relaxation:
            relax_error = _device_thermal_relaxation_error(
                qubits, relax_time, relax_params, temperature,
                thermal_relaxation)

        # Get depolarizing error channel
        if gate_error:
            depol_error = _device_depolarizing_error(
                qubits, error_param, relax_error, standard_gates, warnings=warnings)

        # Combine errors
        if depol_error is None and relax_error is None:
            # No error for this gate
            pass
        elif depol_error is not None and relax_error is None:
            # Append only the depolarizing error
            errors.append((name, qubits, depol_error))
            # Append only the relaxation error
        elif relax_error is not None and depol_error is None:
            errors.append((name, qubits, relax_error))
        else:
            # Append a combined error of depolarizing error
            # followed by a relaxation error
            combined_error = depol_error.compose(relax_error)
            errors.append((name, qubits, combined_error))
    return errors

