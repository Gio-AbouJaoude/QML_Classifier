
import pennylane as qml

num_feat = 4 #<
num_params = 7 #<

num_wires = 2 #<
thread_count = 0 #<

circuit_name = "Beta-Iris Mach 1" #<


device, backend = "qiskit.aer", "qasm_simulator"
dev = qml.device(device, wires = num_wires, backend = backend)

#QML Circuit
@qml.qnode(dev)
def circuit(features, params):
  
  qml.RX(features[0], wires = 0)
  qml.RX(features[1], wires = 1)

  qml.broadcast(qml.CZ, wires = [0, 1], pattern="ring")

  qml.RX(params[0], wires = 0)
  qml.RX(params[1], wires = 1)

  qml.broadcast(qml.CZ, wires = [1, 0], pattern="ring")

  qml.RX(features[2], wires = 0)
  qml.RX(features[3], wires = 1)

  qml.broadcast(qml.CZ, wires = [0, 1], pattern="ring")

  qml.RX(params[2], wires = 0)
  qml.RX(params[3], wires = 1)

  qml.broadcast(qml.CZ, wires = [1, 0], pattern="ring")

  qml.RX(features[0], wires = 0)
  qml.RX(features[1], wires = 1)

  qml.broadcast(qml.CZ, wires = [0, 1], pattern="ring")

  qml.RX(params[4], wires = 0)
  qml.RX(params[5], wires = 1)

  qml.broadcast(qml.CZ, wires = [1, 0], pattern="ring")

  qml.RX(features[2], wires = 0)
  qml.RX(features[3], wires = 1)

  qml.broadcast(qml.CZ, wires = [0, 1], pattern="ring")

  qml.RX(params[6], wires = 0)

  qml.broadcast(qml.CZ, wires = [1, 0], pattern="ring")

  return qml.expval(qml.PauliZ(0))
