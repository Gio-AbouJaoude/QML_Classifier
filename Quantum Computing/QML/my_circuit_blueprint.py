import numpy as np
import pennylane as qml

num_wires = 3
device_name = 'qulacs.simulator'
dev = qml.device(device_name, wires = num_wires)

@qml.qnode(dev)
def even_circuit(features, params):
    # Quantum node for the even circuit
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

    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


@qml.qnode(dev)
def odd_circuit(features, params):
    # Quantum node for the odd circuit
    qml.RX(features[0], wires = 0)
    qml.RX(features[1], wires = 1)
    qml.RX(features[2], wires = 2)

    qml.broadcast(qml.CZ, wires = [0, 1], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [1, 2], pattern = "ring")

    qml.RX(params[0], wires = 0)
    qml.RX(params[1], wires = 1)
    qml.RX(params[2], wires = 2)

    qml.broadcast(qml.CZ, wires = [1, 2], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [2, 0], pattern = "ring")

    qml.RX(features[0], wires = 0)
    qml.RX(features[1], wires = 1)
    qml.RX(features[2], wires = 2)

    qml.broadcast(qml.CZ, wires = [2, 0], pattern = "ring")
    qml.broadcast(qml.CZ, wires = [0, 1], pattern = "ring")

    qml.RX(params[4], wires = 0)
    qml.RX(params[5], wires = 1)
    qml.RX(params[6], wires = 2)

    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))


num_feat = 13
num_params = 46
thread_count = 0
circuit_name = "Network-Wine V_1"
sub_circuits = [{'Circuit Name': 'Even Circuit','Circuit Function': even_circuit, 'Features': 2, 'Parameters': 4, 'Wires': 2},
            {'Circuit Name': 'Odd Circuit','Circuit Function': odd_circuit, 'Features': 3, 'Parameters': 7, 'Wires': 3}]

layer_a = [(even_circuit,2,4), (even_circuit,2,4), (even_circuit,2,4), (even_circuit,2,4), (even_circuit,2,4), (odd_circuit,3, 7)]
layer_b =            [(even_circuit,2,4),                   (even_circuit,2,4),                     (even_circuit,2,4)]
layer_c =                                                   [(odd_circuit,3, 7)]


layers = [layer_a, layer_b, layer_c]



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

# def test_feat(first_layer, num_feat):
#     num_feat_test = 0
#     for circ_val in first_layer:
#         num_feat_test = num_feat_test+ circ_val[1]
#     assert num_feat == num_feat_test

def test_params(layers, num_params):
    # Function to test the number of parameters
    num_params_test = sum(circ_val[2] for layer in layers for circ_val in layer)
    assert num_params == num_params_test

# def test_params(layers, num_params):
#     num_params_test = 0
#     for circ_val in [j for i in layers for j in i]:
#         num_params_test = num_params_test + circ_val[2]
#     assert num_params == num_params_test

test_feat(layers[0], num_feat)
test_params(layers, num_params)


