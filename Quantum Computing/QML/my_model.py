
from my_circuit_blueprint import circuit, circuit_name
from my_training import train_model, quick_train_model
from my_data import q_scale_data, target_data, split_data
from my_metrics import print_df_stats, metrics_test, print_metrics_test
from my_plots import plotly_scatter, make_train_plots, decision_plots, print_fig_dict
from my_qml import make_weights, predict, predict_all, calc_expectations, calc_expectations_all
from my_manager import save_version, save_recordings, load_recordings, get_circuit_settings, get_model_dir


class quantum_model:
  def __init__(self, n_circuits = 3, rng_seed = None, directory = get_model_dir()):
    """
    Initializes a new quantum_model instance.
    Args:
        n_circuits (int): Number of circuits in the model.
        rng_seed (int): Random seed for weight initialization.
        directory (str): Directory for saving model recordings.
    """
    self.rng_seed = rng_seed
    self.n_circuits = n_circuits
    self.circuit_name = circuit_name
    self.settings = get_circuit_settings() if directory else None
    self.weights = make_weights(n_circuits, rng_seed = self.rng_seed)
    self.folder_directory = save_version(directory) if directory else None
    self.w_df, self.m_df, self.e_df, self.b_df, self.plot_dict, self.plot_list = None, None, None, None, {}, []



  def __str__(self):
    """
    Returns the circuit name of the quantum model.
    """
    return self.circuit_name

  def __repr__(self):
    """
    Returns a string representation of the quantum_model.
    """
    return f"Quantum Model: {self.circuit_name}"



  def target_data(self, df, y_col_name):
    """
    Extracts the target variable and feature data from a DataFrame.
    Args:
        df (DataFrame): Input DataFrame containing the target variable and features.
        y_col_name (str): Name of the column representing the target variable.
    Returns:
        tuple: A tuple containing two elements: the feature data and the target data.
    """
    return target_data(df, y_col_name)

  def scale_data(self, df, cat_col = None):
    """
    Scales the data in a DataFrame using min-max scaling.
    Args:
        df (DataFrame): Input DataFrame to be scaled.
        cat_col (list or None): List of column names representing categorical columns.
    Returns:
        DataFrame: Scaled DataFrame.
    """
    return q_scale_data(df, cat_col_name_list = cat_col)

  def split_data(self, X, Y, test_size = 0.15, random_state = None):
    """
    Splits the feature and target data into training and testing sets.
    Args:
        X (array-like): Input feature data.
        Y (array-like): Input target data.
        test_size (float): Proportion of the data to be included in the test set.
        random_state (int): Random seed for reproducibility.
    Returns:
        tuple: A tuple containing four elements: X_train, X_test, Y_train, Y_test.
    """
    return split_data(X, Y, test_size = test_size, random_state = random_state)

  def scale_and_split(self, X, Y, test_size = 0.15, random_state = None):
    """
    Scales the feature data and splits it with the target data into training and testing sets.
    Args:
        X (array-like): Input feature data.
        Y (array-like): Input target data.
        test_size (float): Proportion of the data to be included in the test set.
        random_state (int): Random seed for reproducibility.
    Returns:
        tuple: A tuple containing four elements: X_train, X_test, Y_train, Y_test.
    """
    return split_data(q_scale_data(X), Y, test_size = test_size, random_state = random_state)
  
  
  
  def set_dirirectory(self, directory = None):
    """
    Sets the directory for saving model recordings.
    Args:
        directory (str or None): Directory path. If None, uses the default directory.
    Returns:
        str: The directory path set.
    """
    self.folder_directory = directory if directory else get_model_dir()
    return self.folder_directory

  def refresh_weights(self, rng_seed = None):
    """
    Refreshes the weights of the model's circuits.
    Args:
        rng_seed (int or None): Random seed for weight initialization. If None, uses the current rng_seed value.
    """
    self.rng_seed = rng_seed if rng_seed else self.rng_seed
    self.weights = make_weights(self.n_circuits, rng_seed = rng_seed)
  
  def save_recordings(self, recording_name):
   """
    Saves the model recordings, including weight, metric, expectation, and beta-value recordings, along with associated plots.
    Args:
        recording_name (str): Name of the recording.
    """
   df_list = [self.w_df, self.m_df, self.e_df, self.b_df]
   save_recordings(self.folder_directory, recording_name, self.n_circuits, self.weights, df_list, self.plot_dict, self.plot_list)
  
  def load_recordings(self, recording_name):
   """
    Loads the model recordings from a previously saved version.
    Args:
        recording_name (str): Name of the recording.
    """
   self.n_circuits, self.weights, df_list, self.plot_dict, self.plot_list = load_recordings(self.folder_directory, recording_name)
   self.w_df, self.m_df, self.e_df, self.b_df = df_list




  def circuit(self):
    """
    Returns the circuit object associated with the model.
    """
    return circuit

  def circuit(self, features, parameters):
    """
    Creates and returns a circuit object with the specified features and parameters.
    Args:
        features (array-like): Input features.
        parameters (array-like): Input parameters.
    Returns:
        Circuit: The created circuit object.
    """
    return circuit(features, parameters)

  


  def predict(self, x):
    """
    Makes a single prediction for the given input x using the model's weights.
    Args:
        x (array-like): Input data for prediction.
    Returns:
        array: The predicted output.
    """
    return predict(x, self.weights, self.n_circuits)

  def predict_all(self, X):
    """
    Makes predictions for multiple inputs in the matrix X using the model's weights.
    Args:
        X (array-like): Input data for predictions.
    Returns:
        array: The predicted outputs.
    """
    return predict_all(X, self.weights, self.n_circuits)

  def predict_prob(self, x):
    """
    Calculates the expectation values (probabilities) for the given input x using the model's weights.
    Args:
        x (array-like): Input data for calculating expectation values.
    Returns:
        array: The calculated expectation values.
    """
    return calc_expectations(x, self.weights, self.n_circuits)

  def predict_prob_all(self, X):
    """
    Calculates the expectation values (probabilities) for multiple inputs in the matrix X using the model's weights.
    Args:
        X (array-like): Input data for calculating expectation values.
    Returns:
        array: The calculated expectation values.
    """
    return calc_expectations_all(X, self.weights, self.n_circuits)





  def metric_test(self, X_test, Y_test):
    """
    Prints the classification metrics for the model's predictions on the provided test data.
    Args:
        X_test (array-like): Test feature data.
        Y_test (array-like): Test target data.
    """
    print_metrics_test(None, metrics_test(self.weights, X_test, Y_test, self.n_circuits))

  def quick_fit(self, data_tuple, alpha = 0.1, n_epochs = 10, display = False):
    """
    Trains the model using the provided data in a quick training mode.
    Args:
        data_tuple (tuple): A tuple containing the feature data (X) and target data (Y).
        alpha (float): Learning rate.
        n_epochs (int): Number of training epochs.
        display (bool): Whether to display training progress.
    """
    self.weights = quick_train_model(data_tuple, self.n_circuits, self.weights, alpha = alpha, n_epochs = n_epochs, display = display)

  def fit(self, data_tuple, alpha = 0.1, n_epochs = 10, display = True):
    """
    Trains the model using the provided data in a full training mode.
    Args:
        data_tuple (tuple): A tuple containing the feature data (X) and target data (Y).
        alpha (float): Learning rate.
        n_epochs (int): Number of training epochs.
        display (bool): Whether to display training progress.
    """
    self.weights, df_tuple = train_model(data_tuple, self.n_circuits, self.weights, alpha = alpha, n_epochs = n_epochs, display = display)
    train_plots, train_plot_list = make_train_plots(df_tuple, self.n_circuits)
    self.w_df, self.m_df, self.e_df, self.b_df = df_tuple
    self.plot_list.extend(train_plot_list)
    self.plot_dict.update(train_plots)



  def plotly_scatter(self, x_values, y_values, color_values, title = None, size_values = None):
    """
    Generates a scatter plot using the specified data.
    Args:
        x_values (array-like): x-coordinates of the data points.
        y_values (array-like): y-coordinates of the data points.
        color_values (array-like): Values used for coloring the data points.
        title (str or None): Title of the plot.
        size_values (array-like or None): Values used for sizing the data points.
    Returns:
        plotly.graph_objects.Figure: The generated scatter plot.
    """
    return plotly_scatter(x_values, y_values, color_values, title = title, size_values = size_values)

  def print_stats(self, window_frac = 6):
    """
    Prints statistics about the expectation values recorded during model training.
    Args:
        window_frac (int): Fraction of the data window for calculating statistics.
    """
    print("Expectation Statistics:")
    print_df_stats(self.e_df, window_frac = window_frac) if self.e_df else None

  def print_plot_keys(self):
    """
    Prints the available plot keys for the model.
    """
    print("Model Plot Map:")
    print_fig_dict(self.plot_dict) if self.plot_dict else None

#TODO/UNDER CONSTRUCTION
  # def decision_plots(self, X, Y, delta = 0.99):
  #   """
  #   Generates decision and expectation surface plots using the provided data.
  #   Args:
  #       X (array-like): Feature data.
  #       Y (array-like): Target data.
  #       delta (float): Decision threshold for surface plot generation.
  #   """
  #   dec_plots, dec_plot_list = decision_plots(X, Y, self.weights, delta = delta)
  #   self.plot_list.extend(dec_plot_list)
  #   self.plot_dict.update(dec_plots)

