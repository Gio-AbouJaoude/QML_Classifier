
import numpy as np
import pandas as pd
import plotly.io as pio
from copy import deepcopy
import plotly.express as px
import plotly.graph_objects as go
from pprint import pprint as dict_print
from my_qml import calc_expectations_all, classify_expectations_all

# Set the default renderer for Plotly to "notebook"
pio.renderers.default = "notebook"
# Set the plotting backend for Pandas to Plotly
pd.options.plotting.backend = "plotly"
# Define a color set for plots
color_set = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#9D755D']

def get_subset(df, name_conv, append_col_names = []):
  # Get a subset of columns from the DataFrame `df` based on a naming convention (`name_conv`)
  col_subset_names = []

  for name in df.columns:
    col_subset_names.append(name) if name_conv in name else None
  col_subset_names = col_subset_names + append_col_names
  return df[col_subset_names]

def extract_nested_values(plot_dict):
  # Recursive generator function to extract values from a nested dictionary or a list
  if isinstance(plot_dict, list):
    for sub_dict in plot_dict:
      yield from extract_nested_values(sub_dict)
  elif isinstance(plot_dict, dict):
    for value in plot_dict.values():
      yield from extract_nested_values(value)
  else:
    yield plot_dict


def make_metric_plots(m_df, n_circuits, plot_template):
  # Create plots for metrics based on the provided DataFrame (`m_df`), number of circuits (`n_circuits`), and plot template
  metric_plots, metric_circ_plots, weighted_plots, macro_plots, cross_plots = {}, {}, {}, {}, {}
  # Plot metrics for all circuits
  metric_plots['All'] = m_df.plot(title='Metrics recorded over every optimization', template=plot_template)

  # Extract macro metrics subset from `m_df`
  macro_subset = get_subset(m_df, 'macro avg', append_col_names=['model_accuracy'])
  # Plot macro metrics for all circuits
  macro_plots['All'] = macro_subset.plot(title='Macro metrics recorded over every optimization', template=plot_template)

  # Plot individual macro metrics for all circuits
  for metric_name in ['macro avg_precision', 'macro avg_recall', 'macro avg_f1-score']:
    title_name = metric_name[0].upper() + metric_name[1:]
    macro_plots[metric_name] = get_subset(m_df, metric_name).plot(
      title=f'{title_name} metric recorded over every optimization', template=plot_template)

  # Extract weighted metrics subset from `m_df`
  weighted_subset = get_subset(m_df, 'weighted avg', append_col_names=['model_accuracy'])
  # Plot weighted metrics for all circuits
  weighted_plots['All'] = weighted_subset.plot(title='Weighted metrics recorded over every optimization', template=plot_template)

  # Plot individual weighted metrics for all circuits
  for metric_name in ['weighted avg_precision', 'weighted avg_recall', 'weighted avg_f1-score']:
    title_name = metric_name[0].upper() + metric_name[1:]
    weighted_plots[metric_name] = get_subset(m_df, metric_name).plot(
      title=f'{title_name} metric recorded over every optimization', template=plot_template)

  # Extract all circuit metrics subset from `m_df`
  all_circ_subset = get_subset(m_df, 'C_', append_col_names=['model_accuracy'])
  # Plot all circuit metrics for all circuits
  metric_circ_plots['All'] = all_circ_subset.plot(title='All circuit metrics recorded over every optimization', template=plot_template)

  # Plot individual circuit metrics for each circuit
  for i in range(n_circuits):
    circ_subset = get_subset(m_df, f'C_{i}', append_col_names=['model_accuracy'])
    metric_circ_plots[f'C_{i}'] = circ_subset.plot(
      title=f'Circuit {i} metrics recorded over every optimization', template=plot_template)

  # Plot cross-metrics (precision, recall, f1-score)
  for metric_name in ['precision', 'recall', 'f1-score']:
    title_name = metric_name[0].upper() + metric_name[1:]
    cross_plots[title_name] = get_subset(m_df, metric_name).plot(
      title=f'{title_name} metric recorded over every optimization', template=plot_template)

  # Create a dictionary to store all metric plots
  metric_plots['Macro'] = macro_plots
  metric_plots['Cross'] = cross_plots
  metric_plots['Weighted'] = weighted_plots
  metric_plots['Circuits'] = metric_circ_plots

  return metric_plots

def make_train_plots(df_tuple, n_circuits, frac_window=10):
  # Create plots for training based on the provided DataFrame tuple (`df_tuple`), number of circuits (`n_circuits`), and fraction window
  plot_template = "plotly_dark"
  w_df, m_df, e_df, b_df = df_tuple
  plot_dict, beta_plots, metric_plots, weight_plots, expect_plots, plot_list = {}, {}, {}, {}, {}, []

  # Generate metric plots
  metric_plots = make_metric_plots(m_df, n_circuits, plot_template)

  # Plot weights for all circuits
  weight_plots["All"] = w_df.plot(title='Weights recorded over every optimization', template=plot_template)

  # Plot beta-values for all circuits
  beta_plots["Values"] = b_df.plot(title='Beta-values recorded over every optimization', template=plot_template)

  # Plot expectations for all circuits
  expect_plots["All"] = e_df.plot(title='Expectations recorded over every optimization', template=plot_template)

  # Plot weights and expectations for each circuit
  for i in range(n_circuits):
    weight_plots[f"C_{i}"] = get_subset(w_df, f"C_{i}").plot(title=f"Circuit {i} Parameters over Optimization", template=plot_template)
    expect_plots[f"C_{i}"] = get_subset(e_df, f"C_{i}").plot(title=f"Circuit {i} Expectations over Optimization", template=plot_template)

  # Set the window size for rolling mean and variance
  window = int(len(e_df) * (1 / frac_window))
  # Plot rolling mean of beta-values and expectations
  beta_plots["Mean"] = b_df.rolling(window).mean().plot(title='Beta-values rolling mean with window of ' + str(window), template=plot_template)
  expect_plots["Mean"] = e_df.rolling(window).mean().plot(title='Expectations rolling mean with window of ' + str(window), template=plot_template)
  # Plot rolling variance of beta-values and expectations
  beta_plots["Variance"] = b_df.rolling(window).var().plot(title='Beta-values rolling variance with window of ' + str(window), template=plot_template)
  expect_plots["Variance"] = e_df.rolling(window).var().plot(title='Expectations rolling variance with window of ' + str(window), template=plot_template)

  # Create a dictionary to store all plots
  plot_dict["Beta"] = beta_plots
  plot_dict["Metrics"] = metric_plots
  plot_dict["Weights"] = weight_plots
  plot_dict["Expectations"] = expect_plots

  # Extract all plot values into a list
  for key in plot_dict.keys():
    plot_list = plot_list + list(extract_nested_values(plot_dict[key]))

  return plot_dict, plot_list


def order_plotly_legend(figure):
  # Reorder the legend items in a Plotly figure based on the original order of appearance
  trace = {}
  figure_data = figure.data
  num_targets = len(figure_data)
  sorted_trace = [None]*num_targets

  # Create a dictionary to store the traces
  for i in range(num_targets):
    trace[figure_data[i]['name']] = figure_data[i]

  # Sort the traces based on the original order of appearance
  for i, key in enumerate(sorted(trace)):
    sorted_trace[i] = trace[key]
    sorted_trace[i]['marker']['color'] = color_set[i]

  # Update the figure data with the sorted traces
  figure.data = tuple(sorted_trace)
  return figure

#EXPERIMENTAL/TODO
#--------------------------------------------------------------------------------------------------------------------------------------------------

def plotly_scatter(x_values, y_values, color_values, title = None, size_values = None):
  theme = "plotly_dark"
  color_values = np.array(color_values).astype(str)
  scatter = px.scatter(x = x_values, y = y_values, color = color_values, size = size_values, title = title, template = theme)
  return order_plotly_legend(scatter)


def decision_layout(data, title = None):
  theme = "plotly_dark"
  axis_range = [-1.57, 4.71]
  decision_fig = go.Figure(data = data)
  decision_fig.update_layout(title = title)
  decision_fig.update_layout(template = theme)
  decision_fig.update_layout(xaxis_range = axis_range)
  decision_fig.update_layout(yaxis_range = axis_range)
  return decision_fig

def discrete_colorscale(num_targets):

  color_scale = []
  linear_values = np.linspace(0, 1, num = num_targets+1)

  for i in range(num_targets):
    color_scale.append([linear_values[i], color_set[i]])
    color_scale.append([linear_values[i+1], color_set[i]])

  return color_scale


def plotly_contour(elevation, x_axis, y_axis, title):

  tick_text = None
  tick_values = None
  contour_colorscale = [[0, 'DarkMagenta'], [1, 'Green']]
  color_bar = dict(thickness=25, tickvals = tick_values, ticktext = tick_text)
  contour = go.Contour(z = elevation, x = x_axis, y = y_axis, colorscale = contour_colorscale, contours = dict(showlabels = True), colorbar=color_bar)

  return decision_layout(contour, title = title)


def plotly_heatmap(estimates, grid, n_circuits, title):

  colors = discrete_colorscale(n_circuits)
  tick_text = [i for i in range(n_circuits)]
  spacing_fraction = (n_circuits-1)/(2*n_circuits)
  tick_values = [(1+2*i)*spacing_fraction for i in range(n_circuits)]
  color_bar = dict(thickness = 25, tickvals = tick_values, ticktext = tick_text)
  heatmap = go.Heatmap(x = grid[:, 0], y = grid[:, 1], z = estimates, colorscale = colors, colorbar = color_bar)

  return decision_layout(heatmap, title = title)


def make_meshgrid(x, y, h = 0.05):
  padding = np.pi/2
  x_axis = np.arange(x.min() - padding, x.max() + padding, h)
  y_axis = np.arange(y.min() - padding, y.max() + padding, h)

  x_grid, y_grid = np.meshgrid(x_axis, y_axis)
  cart_grid = np.c_[x_grid.ravel(), y_grid.ravel()]

  return cart_grid, x_axis, y_axis, x_grid.shape


def decision_plots(X, Y, weights, delta = 0.05):
  plot_dict = {}
  X = np.array(X)
  n_circuits = max(Y)+1
  cart_grid, x_axis, y_axis, mesh_shape = make_meshgrid(X[:,0], X[:,1], h = delta)
  all_expectations = np.array( calc_expectations_all(cart_grid, weights, n_circuits) )
  
  for i in range(n_circuits):
    elevation = np.reshape(all_expectations[:, i], mesh_shape)
    plot_dict[f'C_{i}'] = plotly_contour(elevation, x_axis, y_axis, f'C_{i} Circuit Expectation Surface')

  classifications = classify_expectations_all(all_expectations)
  max_elevation = np.reshape(np.amax(all_expectations, axis = 1), mesh_shape)
  plot_dict['Expectation Surface'] = plotly_contour(max_elevation, x_axis, y_axis, 'Model Expectation Surface')
  plot_dict['Decision Surface'] = plotly_heatmap(classifications, cart_grid, n_circuits, 'Model Decision Surface')

  return plot_dict, [*list(plot_dict)]
#--------------------------------------------------------------------------------------------------------------------------------------------------

def alter_dict(d):
  # Recursively modify a dictionary `d` in-place to convert Figure objects to their titles
  if d is not None:
    for key, value in d.items():
      if isinstance(value, dict):
        alter_dict(value)
      if isinstance(value, go.Figure):
        d[key] = value.layout['title']['text']

def print_fig_dict(fig_dict):
  # Print a modified version of the figure dictionary `fig_dict` with Figure objects replaced by their titles
  deep_fig_dict = deepcopy(fig_dict)
  alter_dict(deep_fig_dict)
  dict_print(deep_fig_dict)