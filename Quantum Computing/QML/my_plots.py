
import numpy as np
import pandas as pd
import plotly.io as pio
from copy import deepcopy
import plotly.express as px
import plotly.graph_objects as go
from pprint import pprint as dict_print
from my_qml import calc_expectations_all, classify_expectations_all

pio.renderers.default = "notebook"
pd.options.plotting.backend = "plotly"
color_set = ['#636EFA','#EF553B','#00CC96','#AB63FA','#FFA15A','#19D3F3','#FF6692','#B6E880','#FF97FF','#FECB52','#9D755D']

def get_subset(df, name_conv):
  col_subset_names = []

  for name in df.columns:
    col_subset_names.append(name) if name_conv in name else None

  return df[col_subset_names]

def make_train_plots(df_tuple, n_circuits, frac_window = 10):
  plot_template = "plotly_dark"
  w_df, m_df, e_df, b_df = df_tuple
  metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
  plot_dict, beta_plots, metric_plots, weight_plots, expect_plots, plot_list = {}, {}, {}, {}, {}, []

  
  metric_plots["All"] = m_df.plot(title = 'Metrics recorded over every optimization', template = plot_template)
  weight_plots["All"] = w_df.plot(title = 'Weights recorded over every optimization', template = plot_template)
  beta_plots["Values"] = b_df.plot(title = 'Beta-values recorded over every optimization', template = plot_template)
  expect_plots["All"] = e_df.plot(title = 'Expectations recorded over every optimization', template = plot_template)

  for name in metric_names:
    metric_plots[name] = get_subset(m_df, name).plot(title = f'{name} recorded over every optimization', template = plot_template)

  for i in range(n_circuits):
    weight_plots[f"C_{i}"] = get_subset(w_df, f"C_{i}").plot(title = f"Circuit {i} Parameters over Optimization", template = plot_template)
    expect_plots[f"C_{i}"] = get_subset(e_df, f"C_{i}").plot(title = f"Circuit {i} Expectations over Optimization", template = plot_template)


  window = int( len(e_df)*(1/frac_window) )
  beta_plots["Mean"] = b_df.rolling(window).mean().plot(title = 'Beta-values rolling mean with window of '+str(window), template = plot_template)
  expect_plots["Mean"] = e_df.rolling(window).mean().plot(title = 'Expectations rolling mean with window of '+str(window), template = plot_template)
  beta_plots["Variance"] = b_df.rolling(window).var().plot(title = 'Beta-values rolling variance with window of '+str(window), template = plot_template)
  expect_plots["Variance"] = e_df.rolling(window).var().plot(title = 'Expectations rolling variance with window of '+str(window), template = plot_template)

  plot_dict["Beta"] = beta_plots
  plot_dict["Metrics"] = metric_plots
  plot_dict["Weights"] = weight_plots
  plot_dict["Expectations"] = expect_plots

  plot_list = [*list(metric_plots.values()), *list(weight_plots.values()), *list(expect_plots.values()), *list(beta_plots.values())]

  return plot_dict, plot_list

def order_plotly_legend(figure):
  trace = {}
  figure_data = figure.data
  num_targets = len(figure_data)
  sorted_trace = [None]*num_targets

  for i in range(num_targets):
    trace[figure_data[i]['name']] = figure_data[i]

  for i, key in enumerate(sorted(trace)):
    sorted_trace[i] = trace[key]
    sorted_trace[i]['marker']['color'] = color_set[i]

  figure.data = tuple(sorted_trace)
  return figure

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


def alter_dict(d):
  if d is not None:
    for key, value in d.items():
      if isinstance(value, dict):
        alter_dict(value)
      if isinstance(value, go.Figure):
        d[key] = value.layout['title']['text']

def print_fig_dict(fig_dict):
  deep_fig_dict = deepcopy(fig_dict)
  alter_dict(deep_fig_dict)
  dict_print(deep_fig_dict)