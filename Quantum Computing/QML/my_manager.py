
import os
import pickle
import numpy as np
import pandas as pd
import my_circuit_blueprint
from my_circuit_blueprint import qml, dev, circuit, circuit_name, num_params, num_feat, thread_count, circuit_name, num_wires

os_directory = 'c:\\Users\\gga22\\OneDrive\\Desktop\\Quantum Computing\\Models'


def get_model_dir():
  return os_directory

def get_blueprint_dir():
  return os.path.dirname(os.path.realpath(my_circuit_blueprint.__file__)) + "\\my_circuit_blueprint.py"

def remove_none(df_list):
  return [df for df in df_list if df is not None]



def save_weights(weights, folder_dir):
  os.makedirs(folder_dir + "\\model_weights")
  os.chdir(folder_dir + "\\model_weights")
  weights_np = np.transpose(weights)
  model_weight_df = pd.DataFrame(weights_np)
  model_weight_df.to_csv('weights.txt', header = False, sep = '\t')
  os.chdir(folder_dir)

def zip_df_list(df_list, folder_dir):
  os.makedirs(folder_dir + "\\data")
  os.chdir(folder_dir + "\\data")
  for (df, file_name) in zip(df_list, ["weights", "metrics", "expectations", "beta_values"]):
    df.to_csv(f"{file_name}.csv.zip", index = False, compression = "zip") if df is not None else None
  os.chdir(folder_dir)

def save_plotly_list(plot_list, folder_dir):
  os.makedirs(folder_dir + "\\plots")
  os.chdir(folder_dir + "\\plots")
  for plot in plot_list:
    plot.write_image(f"{plot.layout['title']['text']}.jpeg")
  os.chdir(folder_dir)

def pickle_model(model_tuple, folder_dir):
  os.makedirs(folder_dir + "\\pickled_model")
  os.chdir(folder_dir + "\\pickled_model")
  with open('model.pickle', 'wb') as handle:
    pickle.dump(model_tuple, handle, protocol = pickle.HIGHEST_PROTOCOL)
  os.chdir(folder_dir)

def unpickle_model(folder_dir):
  os.chdir(folder_dir + "\\pickled_model")
  with open('model.pickle', 'rb') as handle:
    model_tuple = pickle.load(handle)
  os.chdir(folder_dir)
  return model_tuple


def save_recordings(dir, recording_name, n_circuits, weights, df_list, plot_dict, plot_list):
  folder_dir = dir + f"\\{recording_name}"
  folder_exists = os.path.exists(folder_dir)
  if not folder_exists:
    os.makedirs(folder_dir)
    os.chdir(folder_dir)
    zip_df_list(df_list, folder_dir)
    save_weights(weights, folder_dir)
    save_plotly_list(plot_list, folder_dir)
    pickle_model((n_circuits, weights, df_list, plot_dict, plot_list), folder_dir)
    os.chdir(dir)
  else:
    print("Warning: Recordings already exist!")

def load_recordings(dir, recording_name):
  folder_dir = dir +f"\\{recording_name}"
  folder_exists = os.path.exists(folder_dir)
  if folder_exists:
    os.chdir(folder_dir)
    model_tuple =  unpickle_model(folder_dir)
    os.chdir(dir)
    return model_tuple
  else:
    print("Warning: Recordings do not exist!")






def get_circuit_settings():
  str_name = f"Model: {circuit_name}"
  str_num_wires = f"Num Wires: {num_wires}"
  str_num_threads = f"Num Threads: {thread_count}"
  str_features = [f"x_{f'{i:02}'}" for i in range(num_feat)]
  str_num_features = f"Num Features: {num_feat} ({str_features[0]})"
  str_parameters = [f"weight_{f'{i:02}'}" for i in range(num_params)]
  str_num_parameters = f"Num Parameters: {num_params} ({str_parameters[0]})"
  str_circuit_diagram = qml.draw(circuit, max_length = 2000)(str_features, str_parameters)
  return f"{str_name}", f"{str_num_wires} || {str_num_threads} || {str_num_features} || {str_num_parameters}", f"{str_circuit_diagram}"

def fill_settings(folder_dir):
  os.chdir(folder_dir)
  with open(folder_dir + "\\circuit_settings.txt", "w", encoding = "utf-8") as file:
    circ_name, circ_settings, circ_diagram = get_circuit_settings()
    print(f"{circ_name}\n{circ_settings}\n{circ_diagram}", file = file)



def get_func_blueprint():
  with open(get_blueprint_dir(), "r", encoding = "utf-8") as file:
    str_code = file.read()
  return str_code

def fill_blueprint(folder_dir):
  os.chdir(folder_dir)
  with open(folder_dir + "\\my_circuit_blueprint.py", "w", encoding = "utf-8") as file:
    print(get_func_blueprint(), file = file)

def version_folder(folder_dir):
  version_folder_dir = folder_dir + "\\version_folder"
  os.makedirs(version_folder_dir)
  os.chdir(version_folder_dir)
  fill_settings(version_folder_dir)
  fill_blueprint(version_folder_dir)
  os.chdir(folder_dir)


def save_version(dir):
  folder_dir = dir + f"\\{circuit_name}"
  folder_exists = os.path.exists(folder_dir)

  if not folder_exists:
    os.makedirs(folder_dir)
    version_folder(folder_dir)
    os.chdir(dir)
  return folder_dir