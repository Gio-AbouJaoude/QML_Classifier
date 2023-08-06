import numpy as np
import pennylane as qml

num_wires = 2
device_name = 'qulacs.simulator'
dev = qml.device(device_name, wires = num_wires)

#QML Circuit
@qml.qnode(dev)
def moon_circuit(features, params):

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


num_feat = 2
num_params = 5
thread_count = 0
circuit_name = "Moon V_4"
sub_circuits = [{'Circuit Name': 'Moon Circuit', 'Circuit Function': moon_circuit, 'Features': 2, 'Parameters': 5, 'Wires': 2}]

layer_0 = [(moon_circuit,2,5)]*1


layers = [layer_0]



def pop_from_list(some_list, num_to_pop):
    # Function to pop values from a list
    return [some_list.pop(0) for i in range(num_to_pop)]

def project_output(circuit_output):
    # Function to project circuit output
    return np.pi*(float(circuit_output) +1)/2

def reverse_projection(circuit_output):
    # Function to reverse-project circuit output
    return circuit_output*2/np.pi -1

def apply_circ(circ_vals, feat_list, params_list):
    # Function to apply a sub-circuit
    circ_func = circ_vals[0]
    num_circ_feats = circ_vals[1]
    num_circ_params = circ_vals[2]
    circ_feats = pop_from_list(feat_list, num_circ_feats)
    circ_params = pop_from_list(params_list, num_circ_params)
    return circ_func(circ_feats, circ_params)

def apply_layers(feat_list, weight_list, layers):
    # Function to apply all layers of the circuit
    layer_count = 0
    layer_input = feat_list.copy()
    params_list = weight_list.copy()
    while layer_count < len(layers):

        layer_circ_vals = layers[layer_count]
        for i in range(len(layer_circ_vals)):
            circ_vals = layer_circ_vals[i]
            circ_output = apply_circ(circ_vals, layer_input, params_list)
            layer_input.append(project_output(circ_output)) if layer_input else layer_input.append(circ_output)
        layer_count += 1
    return layer_input[0]

def circuit(features, params):
    # Main circuit function
    params = list(params)
    features = list(features)
    return apply_layers(features, params, layers)


def test_feat(first_layer, num_feat):
    # Function to test the number of features
    num_feat_test = sum(circ_val[1] for circ_val in first_layer)
    assert num_feat == num_feat_test

def test_params(layers, num_params):
    # Function to test the number of parameters
    num_params_test = sum(circ_val[2] for layer in layers for circ_val in layer)
    assert num_params == num_params_test

test_feat(layers[0], num_feat)
test_params(layers, num_params)
