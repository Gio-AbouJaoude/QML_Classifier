# Quantum Model Class

This is a Python class called `quantum_model` that represents a quantum model for machine learning tasks. It provides various functionalities related to training, prediction, data manipulation, and visualization.

## Class Usage

To use the `quantum_model` class, you need to import it into your Python environment and create an instance of the class.

```python
from quantum_model import quantum_model

# Create an instance of the quantum_model class
model = quantum_model(n_circuits=3, rng_seed=None, directory=None)
```

The `quantum_model` class has the following attributes:

- `n_circuits`: An integer representing the number of circuits in the model.
- `rng_seed`: An optional integer representing the random seed for weight initialization.
- `circuit_name`: A string representing the name of the circuit used in the model.
- `settings`: A dictionary containing the settings of the circuit. This attribute is set when a directory is provided.
- `weights`: A numpy array representing the weights of the model's circuits.
- `folder_directory`: A string representing the directory where model recordings are saved. This attribute is set when a directory is provided.
- `w_df`, `m_df`, `e_df`, `b_df`: DataFrames representing the weight, metric, expectation, and beta-value recordings, respectively.
- `plot_dict`: A dictionary containing plot objects associated with the model.
- `plot_list`: A list of plot objects associated with the model.

The `quantum_model` class provides the following methods:

### Data Handling Methods

- `target_data(df, y_col_name)`: Extracts the target variable and the feature data from a DataFrame.
- `scale_data(df, cat_col=None)`: Scales the data in a DataFrame using min-max scaling. Optionally, categorical columns can be specified.
- `split_data(X, Y, test_size=0.15, random_state=None)`: Splits the feature and target data into training and testing sets.
- `scale_and_split(X, Y, test_size=0.15, random_state=None)`: Scales the feature data and splits it with the target data into training and testing sets.

### Configuration Methods

- `set_directory(directory=None)`: Sets the directory for saving model recordings. If no directory is provided, it uses the default directory.
- `refresh_weights(rng_seed=None)`: Refreshes the weights of the model's circuits. Optionally, a new random seed can be specified.

### Recording Methods

- `save_recordings(recording_name)`: Saves the model recordings, including weight, metric, expectation, and beta-value recordings, along with associated plots.
- `load_recordings(recording_name)`: Loads the model recordings from a previously saved version.

### Circuit Methods

- `circuit()`: Returns the circuit object associated with the model.
- `circuit(features, parameters)`: Creates and returns a circuit object with the specified features and parameters.

### Prediction Methods

- `predict(x)`: Makes a single prediction for the given input `x` using the model's weights.
- `predict_all(X)`: Makes predictions for multiple inputs in the matrix `X` using the model's weights.
- `predict_prob(x)`: Calculates the expectation values (probabilities) for the given input `x` using the model's weights.
- `predict_prob_all(X)`: Calculates the expectation values (probabilities) for multiple inputs in the matrix `X` using the model's weights.

### Training Methods

- `metric_test(X_test, Y_test)`: Prints the classification metrics for the model's predictions on the provided test data.
- `quick_fit(data_tuple, alpha=0.1, n_epochs=10, display=False)`: Trains the model using the provided data in a quick training mode.
- `fit(data_tuple, alpha=0.1, n_epochs=10, display=True)`: Trains the model using the provided data in a full training mode.

### Plotting Methods

- `plotly_scatter(x_values, y_values, color_values, title=None, size_values=None)`: Generates a scatter plot using the specified data.
- `print_stats(window_frac=6)`: Prints statistics about the expectation values recorded during model training.
- `decision_plots(X, Y, delta=0.99)`: Generates decision and expectation surface plots using the provided data.
- `print_plot_keys()`: Prints the available plot keys for the model.