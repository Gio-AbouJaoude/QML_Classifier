
import numpy as np
from multiprocessing.pool import Pool as call_thread_pool
from my_circuit_blueprint import qml, dev, circuit, num_feat, num_params, thread_count # Keep qml & dev

# Calculates the gradient of a single parameter 
def parameter_shift_term(features, params, i):
  shifted = params.copy()
  
  shifted[i] += np.pi/2                 # shift parameter i forward by pi/2
  forward = circuit(features, shifted)  # forward evaluation

  shifted[i] -= np.pi                   # shift parameter i backward by pi/2, (pi/2 - pi = -pi/2)
  backward = circuit(features, shifted) # backward evaluation

  return 0.5* (forward - backward) # the difference between the forward shift and backward shift is twice the gradient

# Calculates the gradients for all parameters in a given circuit
def parameter_shift(features, params):
  gradients = np.zeros([len(params)]) # initialize array of zeros

  for i in range(len(params)):
    gradients[i] = parameter_shift_term(features, params, i) # calculate the gradient for each parameter in the circuit

  return gradients # return the gradients of all parameters in a vector

def thread_parameter_shift(features, params):

  threading_pool = call_thread_pool(processes = thread_count)
  input_list = [(features, params, i) for i in range(len(params))]
  gradients = np.array(threading_pool.starmap(parameter_shift_term, input_list))
  
  return gradients

def quantum_gradient(features, params):
  if thread_count <= 1:
    return parameter_shift(features, params)
  else:
    return thread_parameter_shift(features, params)

# Adjusts all parameters in a given circuit using the gradient to 'optimize' or 'de-optimize'
def step_function(features, params, alpha = 0.1, beta = 1):
  # Each dependent variable will have possible targets that span that variable (possible values of the variable)
  # Each circuit can be 'optimized' to a specific target or 'de-optimized' to a specific target

  # Each parameter is 'optimized' by moving in the positive direction (positive gradient) or 'de-optimized' by...
  # moving in the negative direction (negative gradient)

  params += (alpha*beta)*quantum_gradient(features, params)

  return params

# Helper/utility function for initializing weights at random, can be seeded
def make_weights(n_circuits, rng_seed = None):
  weights = [None]*n_circuits # Initialize weights
  np.random.seed(seed = rng_seed) if rng_seed else None # Seed random number generator for study of anomolies
  
  for i in range(n_circuits):
    weights[i] = 2*np.pi*np.random.random([num_params]) # Generate and append random weights

  return weights # Return random weights

def make_features(rng_seed = None):
  features = [None]*num_feat
  np.random.seed(seed = rng_seed) if rng_seed else None

  for i in range(num_feat):
    features[i] = np.pi*np.random.random() # Generate and append random features

  return features

def calc_expectations(features, weights, n_circuits):
  expectations = [0]*n_circuits # Initialize zeros for classifier
  
  for i in range(n_circuits):
    expectations[i] = (circuit(features, weights[i]) +1)/2 # Get the expectation value from each circuit, one for each target

  return expectations # Return expectations

def calc_expectations_all(all_features, weights, n_circuits):
  all_expectations = [[0]*n_circuits]*len(all_features)

  for i, feature in enumerate(all_features):
    all_expectations[i] = calc_expectations(feature, weights, n_circuits)

  return all_expectations

# TODO
# def thread_calc_expectations_all(all_features, weights, n_circuits, num_threads):
#   threading_pool = call_thread_pool(processes = num_threads)
#   input_list = [(features, params, i) for i in range(len(params))]
#   gradients = np.array(threading_pool.starmap(parameter_shift_term, input_list))
#   return gradients

def classify_expectations(expectations):

  classification = np.argmax(expectations) # Get the index of the largest expectation value, which is taken as the predicted value

  return classification # Return predicted value

def classify_expectations_all(all_expectations):

  classifications = np.argmax(all_expectations, axis = 1)

  return classifications

# Predicts the classification of a dependent variable by using a circuit for each target, this is the classifier
def predict(features, weights, n_circuits):

  expectations = calc_expectations(features, weights, n_circuits = n_circuits) # Get the expectation value from each circuit, one for each target
  classification = classify_expectations(expectations) # Get the index of the largest expectation value, which is taken as the predicted value
  
  return classification # Return predicted value

# Predict all classifications of a given array of features
def predict_all(all_features, weights, n_circuits):

  all_expectations = calc_expectations_all(all_features, weights, n_circuits)
  classifications = classify_expectations_all(all_expectations)

  return classifications # Return classifications