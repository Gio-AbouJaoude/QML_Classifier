
import pennylane as qml

num_feat = 8 #<
num_params = 16 #<

num_wires = 4 #<
thread_count = 0 #<

circuit_name = "Beta-Penguin Mach 1" #<


device, backend = "qiskit.aer", "qasm_simulator"
dev = qml.device(device, wires = num_wires, backend = backend)

#QML Circuit
@qml.qnode(dev)
def circuit(features, params):
  
  qml.RX(features[0], wires = 0)
  qml.RX(features[1], wires = 1)
  qml.RX(features[2], wires = 2)
  qml.RX(features[3], wires = 3)

  qml.broadcast(qml.CZ, wires = [0, 1], pattern = "ring")
  qml.broadcast(qml.CZ, wires = [2, 3], pattern = "ring")

  qml.RX(params[0], wires = 0)
  qml.RX(params[1], wires = 1)
  qml.RX(params[2], wires = 2)
  qml.RX(params[3], wires = 3)

  qml.broadcast(qml.CZ, wires = [1, 2], pattern = "ring")
  qml.broadcast(qml.CZ, wires = [0, 3], pattern = "ring")

  qml.RX(features[4], wires = 0)
  qml.RX(features[5], wires = 1)
  qml.RX(features[6], wires = 2)
  qml.RX(features[7], wires = 3)

  qml.broadcast(qml.CZ, wires = [3, 1], pattern = "ring")
  qml.broadcast(qml.CZ, wires = [2, 0], pattern = "ring")

  qml.RX(params[4], wires = 0)
  qml.RX(params[5], wires = 1)
  qml.RX(params[6], wires = 2)
  qml.RX(params[7], wires = 3)

  qml.broadcast(qml.CZ, wires = [0, 1], pattern = "ring")
  qml.broadcast(qml.CZ, wires = [2, 3], pattern = "ring")

  qml.RX(features[0], wires = 0)
  qml.RX(features[1], wires = 1)
  qml.RX(features[2], wires = 2)
  qml.RX(features[3], wires = 3)

  qml.broadcast(qml.CZ, wires = [1, 2], pattern = "ring")
  qml.broadcast(qml.CZ, wires = [0, 3], pattern = "ring")

  qml.RX(params[8], wires = 0)
  qml.RX(params[9], wires = 1)
  qml.RX(params[10], wires = 2)
  qml.RX(params[11], wires = 3)

  qml.broadcast(qml.CZ, wires = [3, 1], pattern = "ring")
  qml.broadcast(qml.CZ, wires = [2, 0], pattern = "ring")

  qml.RX(features[4], wires = 0)
  qml.RX(features[5], wires = 1)
  qml.RX(features[6], wires = 2)
  qml.RX(features[7], wires = 3)

  qml.broadcast(qml.CZ, wires = [0, 1], pattern = "ring")
  qml.broadcast(qml.CZ, wires = [2, 3], pattern = "ring")

  qml.RX(params[12], wires = 0)
  qml.RX(params[13], wires = 1)
  qml.RX(params[14], wires = 2)
  qml.RX(params[15], wires = 3)

  return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))