
from my_training import train_model, quick_train_model
from my_data import q_scale_data, target_data, split_data
from my_metrics import print_df_stats, metrics_test, print_metrics_test
from my_plots import plotly_scatter, make_train_plots, decision_plots, print_fig_dict
from my_manager import save_version, save_recordings, load_recordings, get_circuit_settings, get_model_dir
from my_qml import make_weights, predict, predict_all, calc_expectations, calc_expectations_all, circuit, circuit_name


class quantum_model:
  def __init__(self, n_circuits = 3, rng_seed = None, directory = get_model_dir()):
    self.rng_seed = rng_seed
    self.n_circuits = n_circuits
    self.circuit_name = circuit_name
    self.folder_directory = save_version(directory)
    _, self.settings, self.blueprint = get_circuit_settings()
    self.weights = make_weights(n_circuits, rng_seed = self.rng_seed)
    self.w_df, self.m_df, self.e_df, self.b_df, self.plot_dict, self.plot_list = None, None, None, None, {}, []



  def __str__(self):
    return self.circuit_name

  def __repr__(self):
    return f"Quantum Model: {self.circuit_name}"



  def target_data(self, df, y_col_name):
    return target_data(df, y_col_name)

  def scale_data(self, df, cat_col = None):
    return q_scale_data(df, cat_col_name_list = cat_col)

  def split_data(self, X, Y, test_size = 0.15, random_state = None):
    return split_data(X, Y, test_size = test_size, random_state = random_state)

  def scale_and_split(self, X, Y, test_size = 0.15, random_state = None):
    return split_data(q_scale_data(X), Y, test_size = test_size, random_state = random_state)



  def refresh_weights(self, rng_seed = None):
    self.rng_seed = rng_seed if rng_seed else self.rng_seed
    self.weights = make_weights(self.n_circuits, rng_seed = rng_seed)


  def save_recordings(self, recording_name):
    df_list = [self.w_df, self.m_df, self.e_df, self.b_df]
    save_recordings(self.folder_directory, recording_name, self.n_circuits, self.weights, df_list, self.plot_dict, self.plot_list)

  def load_recordings(self, recording_name):
    self.n_circuits, self.weights, df_list, self.plot_dict, self.plot_list = load_recordings(self.folder_directory, recording_name)
    self.w_df, self.m_df, self.e_df, self.b_df = df_list




  def circuit(self):
    return circuit

  def circuit(self, features, parameters):
    return circuit(features, parameters)

  


  def predict(self, x):
    return predict(x, self.weights, self.n_circuits)

  def predict_all(self, X):
    return predict_all(X, self.weights, self.n_circuits)

  def predict_prob(self, x):
    return calc_expectations(x, self.weights, self.n_circuits)

  def predict_prob_all(self, X):
    return calc_expectations_all(X, self.weights, self.n_circuits)





  def metric_test(self, X_test, Y_test):
    print_metrics_test(None, metrics_test(self.weights, X_test, Y_test, self.n_circuits))

  def quick_fit(self, data_tuple, alpha = 0.1, n_epochs = 10, display = False):
    self.weights = quick_train_model(data_tuple, self.n_circuits, self.weights, alpha = alpha, n_epochs = n_epochs, display = display)

  def fit(self, data_tuple, alpha = 0.1, n_epochs = 10, display = True):
    self.weights, df_tuple = train_model(data_tuple, self.n_circuits, self.weights, alpha = alpha, n_epochs = n_epochs, display = display)
    train_plots, train_plot_list = make_train_plots(df_tuple, self.n_circuits)
    self.w_df, self.m_df, self.e_df, self.b_df = df_tuple
    self.plot_list.extend(train_plot_list)
    self.plot_dict.update(train_plots)



  def plotly_scatter(self, x_values, y_values, color_values, title = None, size_values = None):
    return plotly_scatter(x_values, y_values, color_values, title = title, size_values = size_values)

  def print_stats(self, window_frac = 6):
    print("Expectation Statistics:")
    print_df_stats(self.e_df, window_frac = window_frac) if self.e_df else None

  def decision_plots(self, X, Y, delta = 0.99):
    dec_plots, dec_plot_list = decision_plots(X, Y, self.weights, delta = delta)
    self.plot_list.extend(dec_plot_list)
    self.plot_dict.update(dec_plots)

  def print_plot_keys(self):
    print("Model Plot Map:")
    print_fig_dict(self.plot_dict) if self.plot_dict else None