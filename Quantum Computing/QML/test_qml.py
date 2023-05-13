
import my_qml
import my_model
import my_plots
import my_metrics
import my_training
import my_circuit_blueprint

# pytest -p no:warnings
# cd "OneDrive\Desktop\Quantum Computing\QML"
# C:\Users\gga22\OneDrive\Desktop\Quantum Computing\QML

def test_circuit():
    num_circuit = 2
    features = my_qml.make_features()
    weights = my_qml.make_weights(num_circuit)[0]
    my_qml.circuit(features, weights)
    assert True


def test_weights():
    num_circuit = 5
    weights = my_qml.make_weights(num_circuit)
    num_params =  my_circuit_blueprint.num_params

    assert len(weights[0]) == num_params
    assert my_training.get_weight_dim(weights)[0] == num_circuit

def test_scale_data():
    x = [[0, 5, 10]]
    scaled_x = my_training.q_scale_data(x)[0]

    assert scaled_x[0] == 0
    assert scaled_x[1] == scaled_x[2]/2

def test_predict():
    n_circuits = 3
    features = my_qml.make_features()
    weights = my_qml.make_weights(n_circuits)
    my_qml.predict(features, weights, n_circuits)
    assert True

def test_model():
    my_model.q_model(n_circuits = 2, rng_seed = 42)
    assert True

def test_quick_train():
    n_circuits = 2
    weights = my_qml.make_weights(n_circuits)
    x_test, y_test = [my_qml.make_features()], [0]
    x_train, y_train = [my_qml.make_features()], [1]
    data_tuple = (x_train, x_test, y_train, y_test)
    my_training.quick_train_model(data_tuple, weights, alpha = 0.1, n_epochs = 1, display = True)
    assert True

def test_dec_plot():
    if my_qml.num_feat == 2:
        n_circuits = 2
        weights = my_qml.make_weights(n_circuits)
        X, Y = [my_qml.make_features(), my_qml.make_features()], [0, 1]
        my_plots.decision_plots(X, Y, weights, delta = 0.99)
    assert True

def test_long_train():
    n_circuits = 2
    weights = my_qml.make_weights(n_circuits)
    x_test, y_test = [my_qml.make_features()], [0]
    x_train, y_train = [my_qml.make_features()], [1]
    data_tuple = (x_train, x_test, y_train, y_test)
    weights, w_df, m_df, e_df = my_training.train_model(data_tuple, weights, alpha = 0.1, n_epochs = 1, display = True)
    
    train_plots = my_plots.make_train_plots((w_df, m_df, e_df))

    my_metrics.print_df_stats(e_df)
    my_plots.print_fig_dict(train_plots)
    assert True