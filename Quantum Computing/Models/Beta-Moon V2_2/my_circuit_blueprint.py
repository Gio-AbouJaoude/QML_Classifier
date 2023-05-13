
import pennylane as qml

num_feat = 2
num_params = 5

num_wires = 2
thread_count = 0

circuit_name = "Beta-Moon V2_2"
dev = qml.device("qiskit.aer", wires = num_wires, backend = 'qasm_simulator')

#['aer_simulator', 'aer_simulator_statevector', 'aer_simulator_density_matrix', 
# 'aer_simulator_stabilizer', 'aer_simulator_matrix_product_state', 'aer_simulator_extended_stabilizer', 
# 'aer_simulator_unitary', 'aer_simulator_superop', 'qasm_simulator', 'statevector_simulator', 'unitary_simulator', 'pulse_simulator']

#QML Circuit
@qml.qnode(dev)
def circuit(features, params):

    qml.RX(features[0], wires = 0)
    qml.RX(features[1], wires = 1)

    qml.broadcast(qml.CZ, wires = [0, 1], pattern = "ring")

    qml.RX(params[0], wires = 0)
    qml.RX(params[1], wires = 1)

    qml.broadcast(qml.CZ, wires = [1, 0], pattern = "ring")

    qml.RX(features[0], wires = 0)
    qml.RX(features[1], wires = 1)

    qml.broadcast(qml.CZ, wires = [0, 1], pattern = "ring")

    qml.RX(params[2], wires = 0)
    qml.RX(params[3], wires = 1)

    qml.broadcast(qml.CZ, wires = [1, 0], pattern = "ring")

    qml.RX(features[0], wires = 0)
    qml.RX(features[1], wires = 1)

    qml.broadcast(qml.CZ, wires = [0, 1], pattern = "ring")

    qml.RX(params[4], wires = 0)

    qml.broadcast(qml.CZ, wires = [1, 0], pattern = "ring")

    return qml.expval(qml.PauliZ(0))
