import numpy as np
import pandas as pd
import time as timer
from my_circuit_blueprint import num_params
from my_qml import step_function, calc_expectations
from my_metrics import metrics_test, print_metrics_test

# --- Possible Beta Functions ---
# expectations[target] - 1 # Delta 
# sum(expectations)/n_circuits - 1 # Theta
# expectations[target]/n_circuits - 1 # Rho
# expectations[target]/sum(expectations) - 1 # Classic
# (sum(expectations) - expectations[target])/n_circuits - 1 # Tau

# Detect and 'optimize'/'de-optimize' the appropriate circuits, optimizing the entire classifier
def optimize_model(features, target, weights, n_circuits, alpha = 0.1):

  weights[target] = step_function(features, weights[target], alpha = alpha) # Preform standard gradient function on circuit associated with target
  expectations = calc_expectations(features, weights, n_circuits)
  beta = expectations[target]/sum(expectations) - 1 # Classic
  
  for j in range(n_circuits):
    if j != target: # Determine which indices are not the current target
      weights[j] = step_function(features, weights[j], alpha = alpha, beta = beta) # Preform negative gradient function on other circuits

  return weights, beta

# The three functions below are for storing values in large arrays, will later be recoded to save in a file (json or simple array txt)

# Record the weights of the classifier in an array
def get_weight_record(weights):
  return list(np.concatenate(weights))

# Record the metrics of the classifier in an array
def get_metric_record(metrics):
  return [metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1-Score']]

def blank_matrix(n_rows, n_cols):
  return np.zeros([n_rows, n_cols], dtype = float)

# Quickly train the model without recording values for speed
def quick_train_model(data_tuple, n_circuits, weights, alpha = 0.1, n_epochs = 10, display = True):

  n = -1
  X_train, X_test, Y_train, Y_test = data_tuple

  # Train for several epochs, each epoch is going through the training data set once
  start_time = timer.time()
  for n in range(n_epochs):
    metrics = metrics_test(weights, X_test, Y_test, n_circuits)
    print_metrics_test(n, metrics, time_diff = timer.time() - start_time) if display else None # Printing is optional

    start_time = timer.time()
    # Iterate through the training data set
    for i in range(len(X_train)):
      # Optimize the classifier
      weights, _, = optimize_model(X_train[i], Y_train[i], weights, n_circuits, alpha = alpha)
      #print("--- %s seconds ---" % (time.time() - start_time))
  
  metrics = metrics_test(weights, X_test, Y_test, n_circuits)
  print_metrics_test(n+1, metrics, time_diff = timer.time() - start_time) if display else None # Printing is optional

  return weights

# Train the classifier by selecting which circuit to 'optimize' and which circuits to 'de-optimize' also records all weights and metrics in arrays
def train_model(data_tuple, n_circuits, weights, alpha = 0.1, n_epochs = 10, display = True):

  n = -1
  row_num = 0
  num_metric_values = 4
  X_train, X_test, Y_train, Y_test = data_tuple


  beta_names = ["Beta Values"]
  expect_names = [f"C_{i}_expect" for i in range(n_circuits)]
  metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
  weight_names = [f"C_{i}_w_{j}" for i in range(n_circuits) for j in range(num_params)]

  start_time = timer.time()
  train_data_length = len(X_train)
  record_length = train_data_length*n_epochs +1
  metrics = metrics_test(weights, X_test, Y_test, n_circuits)
  
  beta_data = blank_matrix(record_length, 1)
  expect_data = blank_matrix(record_length, n_circuits)
  metric_data = blank_matrix(record_length, num_metric_values)
  weight_data = blank_matrix(record_length, n_circuits*num_params)
  

  # Record the weights and metrics for the class in array form, this and every other recording can be recoded to write to an external file
  beta_data[row_num] = 0
  expect_data[row_num] = [0]*n_circuits
  metric_data[row_num] = get_metric_record(metrics)
  weight_data[row_num] = get_weight_record(weights)

  # Train for several epochs, each epoch is going through the training data set once
  for n in range(n_epochs):
    
    print_metrics_test(n, metrics, time_diff = timer.time() - start_time) if display else None # Printing is optional

    start_time = timer.time()
    # Iterate through the training data set
    for i in range(train_data_length):

      row_num = row_num +1

      # Optimize the classifier
      weights, beta_value = optimize_model(X_train[i], Y_train[i], weights, n_circuits, alpha = alpha)
      expectations = calc_expectations(X_train[i], weights, n_circuits = n_circuits)
      metrics = metrics_test(weights, X_test, Y_test, n_circuits)

      beta_data[row_num] = beta_value
      expect_data[row_num] = expectations
      metric_data[row_num] = get_metric_record(metrics)
      weight_data[row_num] = get_weight_record(weights)

  print_metrics_test(n+1, metrics, time_diff = timer.time() - start_time) if display else None # Printing is optional

  # Convert the records of the weights and metrics to dataframe form for easier study, can be easily commented out

  b_df = pd.DataFrame(beta_data, columns = beta_names)
  e_df = pd.DataFrame(expect_data, columns = expect_names)
  m_df = pd.DataFrame(metric_data, columns = metric_names)
  w_df = pd.DataFrame(weight_data, columns = weight_names)

  return weights, (w_df, m_df, e_df, b_df) # Returns the final 'trained' weights and dataframes