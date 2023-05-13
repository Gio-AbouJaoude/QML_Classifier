
import pennylane as qml

num_feat = 32
num_params = 56

num_wires = 16
thread_count = 2

circuit_name = "Delta-Digits V1"
dev = qml.device("qiskit.aer", wires = num_wires, backend = 'qasm_simulator')

#['aer_simulator', 'aer_simulator_statevector', 'aer_simulator_density_matrix', 
# 'aer_simulator_stabilizer', 'aer_simulator_matrix_product_state', 'aer_simulator_extended_stabilizer', 
# 'aer_simulator_unitary', 'aer_simulator_superop', 'qasm_simulator', 'statevector_simulator', 'unitary_simulator', 'pulse_simulator']

#QML Circuit
@qml.qnode(dev, diff_method = "parameter-shift")
def circuit(features, params):

    qml.RX(features[0], wires = 0)
    qml.RX(features[1], wires = 1)
    qml.RX(features[2], wires = 2)
    qml.RX(features[3], wires = 3)

    qml.RX(features[4], wires = 4)
    qml.RX(features[5], wires = 5)
    qml.RX(features[6], wires = 6)
    qml.RX(features[7], wires = 7)

    qml.RX(features[8], wires = 8)
    qml.RX(features[9], wires = 9)
    qml.RX(features[10], wires = 10)
    qml.RX(features[11], wires = 11)

    qml.RX(features[12], wires = 12)
    qml.RX(features[13], wires = 13)
    qml.RX(features[14], wires = 14)
    qml.RX(features[15], wires = 15)

    qml.broadcast(qml.CZ, wires = [0, 1], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [2, 3], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [4, 5], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [6, 7], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [8, 9], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [10, 11], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [12, 13], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [14, 15], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [15, 0], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [3, 4], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [7, 8], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [11, 12], pattern = "ring")

    qml.RX(params[0], wires = 0)
    qml.RX(params[1], wires = 1)
    qml.RX(params[2], wires = 2)
    qml.RX(params[3], wires = 3)

    qml.RX(params[4], wires = 4)
    qml.RX(params[5], wires = 5)
    qml.RX(params[6], wires = 6)
    qml.RX(params[7], wires = 7)

    qml.RX(params[8], wires = 8)
    qml.RX(params[9], wires = 9)
    qml.RX(params[10], wires = 10)
    qml.RX(params[11], wires = 11)

    qml.RX(params[12], wires = 12)
    qml.RX(params[13], wires = 13)
    qml.RX(params[14], wires = 14)
    qml.RX(params[15], wires = 15)

    qml.broadcast(qml.CZ, wires = [1, 0], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [3, 2], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [5, 4], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [7, 6], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [9, 8], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [11, 10], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [13, 12], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [15, 14], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [0, 15], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [4, 3], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [8, 7], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [12, 11], pattern = "ring")

    qml.RX(features[16], wires = 0)
    qml.RX(features[17], wires = 1)
    qml.RX(features[18], wires = 2)
    qml.RX(features[19], wires = 3)

    qml.RX(features[20], wires = 4)
    qml.RX(features[21], wires = 5)
    qml.RX(features[22], wires = 6)
    qml.RX(features[23], wires = 7)

    qml.RX(features[24], wires = 8)
    qml.RX(features[25], wires = 9)
    qml.RX(features[26], wires = 10)
    qml.RX(features[27], wires = 11)

    qml.RX(features[28], wires = 12)
    qml.RX(features[29], wires = 13)
    qml.RX(features[30], wires = 14)
    qml.RX(features[31], wires = 15)

    qml.broadcast(qml.CZ, wires = [0, 1], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [2, 3], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [4, 5], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [6, 7], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [8, 9], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [10, 11], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [12, 13], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [14, 15], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [15, 0], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [3, 4], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [7, 8], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [11, 12], pattern = "ring")

    qml.RX(params[16], wires = 0)
    qml.RX(params[17], wires = 1)
    qml.RX(params[18], wires = 2)
    qml.RX(params[19], wires = 3)

    qml.RX(params[20], wires = 4)
    qml.RX(params[21], wires = 5)
    qml.RX(params[22], wires = 6)
    qml.RX(params[23], wires = 7)

    qml.RX(params[24], wires = 8)
    qml.RX(params[25], wires = 9)
    qml.RX(params[26], wires = 10)
    qml.RX(params[27], wires = 11)

    qml.RX(params[28], wires = 12)
    qml.RX(params[29], wires = 13)
    qml.RX(params[30], wires = 14)
    qml.RX(params[31], wires = 15)

    qml.broadcast(qml.CZ, wires = [1, 0], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [3, 2], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [5, 4], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [7, 6], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [9, 8], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [11, 10], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [13, 12], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [15, 14], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [0, 15], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [4, 3], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [8, 7], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [12, 11], pattern = "ring")

    qml.RX(features[0], wires = 0)
    qml.RX(features[1], wires = 1)
    qml.RX(features[2], wires = 2)
    qml.RX(features[3], wires = 3)

    qml.RX(features[4], wires = 4)
    qml.RX(features[5], wires = 5)
    qml.RX(features[6], wires = 6)
    qml.RX(features[7], wires = 7)

    qml.RX(features[8], wires = 8)
    qml.RX(features[9], wires = 9)
    qml.RX(features[10], wires = 10)
    qml.RX(features[11], wires = 11)

    qml.RX(features[12], wires = 12)
    qml.RX(features[13], wires = 13)
    qml.RX(features[14], wires = 14)
    qml.RX(features[15], wires = 15)

    qml.broadcast(qml.CZ, wires = [0, 1], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [2, 3], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [4, 5], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [6, 7], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [8, 9], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [10, 11], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [12, 13], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [14, 15], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [15, 0], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [3, 4], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [7, 8], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [11, 12], pattern = "ring")

    qml.RX(params[32], wires = 0)
    qml.RX(params[33], wires = 1)
    qml.RX(params[34], wires = 2)
    qml.RX(params[35], wires = 3)

    qml.RX(params[36], wires = 4)
    qml.RX(params[37], wires = 5)
    qml.RX(params[38], wires = 6)
    qml.RX(params[39], wires = 7)

    qml.RX(params[40], wires = 8)
    qml.RX(params[41], wires = 9)
    qml.RX(params[42], wires = 10)
    qml.RX(params[43], wires = 11)

    qml.RX(params[44], wires = 12)
    qml.RX(params[45], wires = 13)
    qml.RX(params[46], wires = 14)
    qml.RX(params[47], wires = 15)

    qml.broadcast(qml.CZ, wires = [1, 0], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [3, 2], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [5, 4], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [7, 6], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [9, 8], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [11, 10], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [13, 12], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [15, 14], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [0, 15], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [4, 3], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [8, 7], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [12, 11], pattern = "ring")

    qml.RX(features[16], wires = 0)
    qml.RX(features[17], wires = 1)
    qml.RX(features[18], wires = 2)
    qml.RX(features[19], wires = 3)

    qml.RX(features[20], wires = 4)
    qml.RX(features[21], wires = 5)
    qml.RX(features[22], wires = 6)
    qml.RX(features[23], wires = 7)

    qml.RX(features[24], wires = 8)
    qml.RX(features[25], wires = 9)
    qml.RX(features[26], wires = 10)
    qml.RX(features[27], wires = 11)

    qml.RX(features[28], wires = 12)
    qml.RX(features[29], wires = 13)
    qml.RX(features[30], wires = 14)
    qml.RX(features[31], wires = 15)

    qml.broadcast(qml.CZ, wires = [0, 1], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [2, 3], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [4, 5], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [6, 7], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [8, 9], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [10, 11], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [12, 13], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [14, 15], pattern = "ring")

    qml.RX(params[48], wires = 0)
    qml.RX(params[49], wires = 2)
    qml.RX(params[50], wires = 4)
    qml.RX(params[51], wires = 6)

    qml.RX(params[52], wires = 8)
    qml.RX(params[53], wires = 10)
    qml.RX(params[54], wires = 12)
    qml.RX(params[55], wires = 14)

    qml.broadcast(qml.CZ, wires = [1, 0], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [3, 2], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [5, 4], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [7, 6], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [9, 8], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [11, 10], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [13, 12], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [15, 14], pattern = "ring")

    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2) @ qml.PauliZ(4) @ qml.PauliZ(6) @ qml.PauliZ(8) @ qml.PauliZ(10) @ qml.PauliZ(12) @ qml.PauliZ(14))