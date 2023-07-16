
import os
import pickle
import numpy as np
import pandas as pd
import my_circuit_blueprint
from my_circuit_blueprint import qml, dev, sub_circuits, circuit_name, num_params, num_feat, thread_count, num_wires, layers # Keep qml & dev

os_directory = os.getcwd()


def get_model_dir():
  # Returns the directory path for the "Models" folder
  return os.path.dirname(os_directory) +'\\Models'

def get_blueprint_dir():
  # Returns the directory path for the "my_circuit_blueprint.py" file
  return os.path.dirname(os.path.realpath(my_circuit_blueprint.__file__)) + "\\my_circuit_blueprint.py"

def remove_none(df_list):
  # Removes `None` values from a list of dataframes and returns a new list without the `None` values
  return [df for df in df_list if df is not None]

def save_weights(weights, folder_dir):
  # Creates a folder named "model_weights" in the given `folder_dir` directory
  os.makedirs(folder_dir + "\\model_weights")
  # Changes the current working directory to the "model_weights" folder
  os.chdir(folder_dir + "\\model_weights")
  # Transposes the `weights` array using `np.transpose()`
  weights_np = np.transpose(weights)
  # Converts the transposed array to a DataFrame
  model_weight_df = pd.DataFrame(weights_np)
  # Saves the DataFrame as a tab-separated values (tsv) file named "weights.txt" in the "model_weights" folder
  model_weight_df.to_csv('weights.txt', header=False, sep='\t')
  # Changes the current working directory back to `folder_dir`
  os.chdir(folder_dir)

def zip_df_list(df_list, folder_dir):
  # Creates a folder named "data" in the given `folder_dir` directory
  os.makedirs(folder_dir + "\\data")
  # Changes the current working directory to the "data" folder
  os.chdir(folder_dir + "\\data")
  for (df, file_name) in zip(df_list, ["weights", "metrics", "expectations", "beta_values"]):
    # For each DataFrame in `df_list`, it saves the DataFrame as a compressed zip file with the file name "{file_name}.csv.zip" in the "data" folder
    # If the DataFrame is `None`, it skips saving it
    df.to_csv(f"{file_name}.csv.zip", index=False, compression="zip") if df is not None else None
  # Changes the current working directory back to `folder_dir`
  os.chdir(folder_dir)

def save_plotly_list(plot_list, folder_dir):
  # Creates a folder named "plots" in the given `folder_dir` directory
  os.makedirs(folder_dir + "\\plots")
  # Changes the current working directory to the "plots" folder
  os.chdir(folder_dir + "\\plots")
  for plot in plot_list:
    # Saves each plot as a JPEG image with the plot title as the file name in the "plots" folder
    plot.write_image(f"{plot.layout['title']['text']}.jpeg")
  # Changes the current working directory back to `folder_dir`
  os.chdir(folder_dir)

def pickle_model(model_tuple, folder_dir):
  # Creates a folder named "pickled_model" in the given `folder_dir` directory
  os.makedirs(folder_dir + "\\pickled_model")
  # Changes the current working directory to the "pickled_model" folder
  os.chdir(folder_dir + "\\pickled_model")
  with open('model.pickle', 'wb') as handle:
    # Pickles and saves the `model_tuple` as a binary file named "model.pickle" in the "pickled_model" folder
    pickle.dump(model_tuple, handle, protocol=pickle.HIGHEST_PROTOCOL)
  # Changes the current working directory back to `folder_dir`
  os.chdir(folder_dir)

def unpickle_model(folder_dir):
  # Changes the current working directory to the "pickled_model" folder in the given `folder_dir`
  os.chdir(folder_dir + "\\pickled_model")
  with open('model.pickle', 'rb') as handle:
    # Loads the pickled model tuple from the "model.pickle" file
    model_tuple = pickle.load(handle)
  # Changes the current working directory back to `folder_dir`
  os.chdir(folder_dir)
  # Returns the loaded model tuple
  return model_tuple

def save_recordings(dir, recording_name, n_circuits, weights, df_list, plot_dict, plot_list):
  # Creates a folder with the given `recording_name` inside the `dir` directory
  folder_dir = dir + f"\\{recording_name}"
  # Checks if the folder already exists
  folder_exists = os.path.exists(folder_dir)
  if not folder_exists:
    # If the folder does not exist, it creates the folder and proceeds with saving the recordings
    os.makedirs(folder_dir)
    # Changes the current working directory to the newly created folder
    os.chdir(folder_dir)
    # Calls the `zip_df_list()`, `save_weights()`, `save_plotly_list()`, and `pickle_model()` functions to save the provided data and plots in the folder
    zip_df_list(df_list, folder_dir)
    save_weights(weights, folder_dir)
    save_plotly_list(plot_list, folder_dir)
    pickle_model((n_circuits, weights, df_list, plot_dict, plot_list), folder_dir)
    # Changes the current working directory back to `dir`
    os.chdir(dir)
  else:
    print("Warning: Recordings already exist!")

def load_recordings(dir, recording_name):
  # Creates the folder path for the given `recording_name` inside the `dir` directory
  folder_dir = dir + f"\\{recording_name}"
  # Checks if the folder exists
  folder_exists = os.path.exists(folder_dir)
  if folder_exists:
    # If the folder exists, it changes the current working directory to the folder
    os.chdir(folder_dir)
    # Calls the `unpickle_model()` function to load the model tuple from the "model.pickle" file in the folder
    model_tuple = unpickle_model(folder_dir)
    # Changes the current working directory back to `dir`
    os.chdir(dir)
    # Returns the loaded model tuple
    return model_tuple
  else:
    print("Warning: Recordings do not exist!")

def get_sub_circuit_settings(sub_circ_dict):
  # Extracts the number of wires, features, and parameters from the sub_circuit_dict
  sub_circ_num_wires = sub_circ_dict['Wires']
  sub_circ_num_feat = sub_circ_dict['Features']
  sub_circ_num_params = sub_circ_dict['Parameters']

  # Creates strings for the number of wires, features, and parameters
  num_wires_str = f"Num Wires: {sub_circ_num_wires}"
  num_features_str = f"Num Features: {sub_circ_num_feat} (X_00)"
  num_parameters_str = f"Num Parameters: {sub_circ_num_params} (Weight_00)"

  # Generates strings for feature names (X_00) and parameter names (Weight_00)
  features_str = [f"X_{f'{i:02}'}" for i in range(sub_circ_num_feat)]
  parameters_str = [f"Weight_{f'{i:02}'}" for i in range(sub_circ_num_params)]

  # Creates a string for the sub-circuit name and its corresponding circuit diagram
  sub_circ_name = f"Sub-Circuit Name: {sub_circ_dict['Circuit Name']}"
  sub_circuit_diagram = qml.draw(sub_circ_dict['Circuit Function'], max_length=2000)(features_str, parameters_str)

  # Returns a formatted string containing the sub-circuit information
  return f"{sub_circ_name}\n{num_wires_str} ||{num_features_str} || {num_parameters_str}\n{sub_circuit_diagram}"

def get_circuit_settings():
  # Generates strings for the model name, number of wires, features, parameters, and thread count
  name_str = f"Model: {circuit_name}"
  num_wires_str = f"Num Wires: {num_wires}"
  num_feats_str = f"Num Features: {num_feat}"
  num_params_str = f"Num Parameters: {num_params}"
  num_threads_str = f"Num Threads: {thread_count}"

  # Returns a formatted string containing the circuit settings
  return f"{name_str}\n{num_threads_str} || {num_wires_str} || {num_feats_str} || {num_params_str}"

def get_model_arch():
  tab = 0
  model_arch_str = ''
  num_layers = len(layers)
  for layer in layers:
    layer_str = ''
    model_arch_str += ' '*tab +'['
    for i in range(len(layer)):
      circuit, features, params = layer[i]
      circuit_str = f"({circuit.__name__}, {features}, {params})"
      model_arch_str += circuit_str
      layer_str += circuit_str
    model_arch_str += ']\n'
    tab += len(layer_str)//num_layers
  return model_arch_str

def fill_settings(folder_dir):
  # Changes the current working directory to `folder_dir`
  os.chdir(folder_dir)
  with open(folder_dir + "\\circuit_settings.txt", "w", encoding="utf-8") as file:
    # Gets the circuit settings string
    circuit_settings = get_circuit_settings()
    # Writes the circuit settings string to the "circuit_settings.txt" file
    print(circuit_settings, file=file)
    # Adds a separator line to the file
    print('-' * 180, file=file)
    for sub_circ_dict in sub_circuits:
      # Gets the sub-circuit settings string
      sub_circuit_settings = get_sub_circuit_settings(sub_circ_dict)
      # Writes the sub-circuit settings string to the file
      print(sub_circuit_settings, file=file)
      # Adds a separator line to the file
      print('-' * 180, file=file)
    # Writes the model architecture string to the file
    print("Model Architecture:", file=file)
    print(get_model_arch(), file=file)
    # Adds a separator line to the file
    print('-' * 180, file=file)

def get_func_blueprint():
  with open(get_blueprint_dir(), "r", encoding="utf-8") as file:
    # Reads the content of the "my_circuit_blueprint.py" file
    str_code = file.read()
  # Returns the content of the file as a string
  return str_code

def fill_blueprint(folder_dir):
  # Changes the current working directory to `folder_dir`
  os.chdir(folder_dir)
  with open(folder_dir + "\\my_circuit_blueprint.py", "w", encoding="utf-8") as file:
    # Writes the content of the "my_circuit_blueprint.py" file to the "my_circuit_blueprint.py" file in the folder
    print(get_func_blueprint(), file=file)

def version_folder(folder_dir):
  # Creates a sub-folder named "version_folder" in the given `folder_dir` directory
  version_folder_dir = folder_dir + "\\version_folder"
  os.makedirs(version_folder_dir)
  # Changes the current working directory to the "version_folder" folder
  os.chdir(version_folder_dir)
  # Fills the settings and blueprint files in the "version_folder" folder
  fill_settings(version_folder_dir)
  fill_blueprint(version_folder_dir)
  # Changes the current working directory back to `folder_dir`
  os.chdir(folder_dir)

def save_version(dir):
  # Creates a folder with the circuit name inside the given `dir` directory
  folder_dir = dir + f"\\{circuit_name}"
  # Checks if the folder already exists
  folder_exists = os.path.exists(folder_dir)

  if not folder_exists:
    # If the folder does not exist, it creates the folder and proceeds with saving the version
    os.makedirs(folder_dir)
    # Fills the version folder with settings and blueprint files
    version_folder(folder_dir)
    # Changes the current working directory back to `dir`
    os.chdir(dir)
  return folder_dir