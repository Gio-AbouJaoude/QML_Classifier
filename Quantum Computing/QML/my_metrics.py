
from numpy import unique
from pandas import DataFrame
from my_qml import predict_all
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Generate standard classification metrics in a dictionary
def metrics_test(weights, X_test, Y_test, n_circuits, method = 'weighted'):

  Y_estimate = predict_all(X_test, weights, n_circuits) # Estimate dependent variable for every row in the test set

  # The estimated variable and the actual variable are used to calculate/generate metrics for the classifier

  metrics = {}
  metrics['Accuracy'] = accuracy_score(Y_test, Y_estimate)
  metrics['ConfMatrix'] = confusion_matrix(Y_test, Y_estimate)
  metrics['F1-Score'] = f1_score(Y_test, Y_estimate, average = method)
  metrics['Recall'] = recall_score(Y_test, Y_estimate, average = method)
  metrics['Precision'] = precision_score(Y_test, Y_estimate, average = method, labels = unique(Y_estimate))
  
  return metrics # Return dictionary of metrics

def percent(value):
  return str(round(value*100, 2)) +'%'

# Helper/utility function for printing metrics dictionary
def print_metrics_test(epoch, metrics, time_diff = None):

  f1 = percent(metrics['F1-Score'])
  recall = percent(metrics['Recall'])
  accuracy = percent(metrics['Accuracy'])
  precision = percent(metrics['Precision'])
  
  width = len(metrics['ConfMatrix'][0])
  index_vec = ['true: ' + str(i) for i in range(width)]
  columns_vec = ['pred: ' + str(i) for i in range(width)]
  conf_mtx = DataFrame(metrics['ConfMatrix'], index = index_vec, columns = columns_vec)

  print("**************************************************************************") if epoch == 0 else None
  print(f"                               At epoch : {epoch}                        ")
  print("--------------------------------------------------------------------------")
  print(f"Accuracy: {accuracy}  |  Precision: {precision}  |  Recall: {recall}  |  F1: {f1}")
  print("Confusion Matrix: ")
  print(conf_mtx)
  if time_diff:
    print("--------------------------------------------------------------------------")
  else:
     print("**************************************************************************")
  print(f"                      Epoch time : {round(time_diff, 2)} seconds         ") if time_diff else None
  print("**************************************************************************") if time_diff else None

# Calculate statistics for a one dimensional array of data
def get_col_stats(df_col, window_frac = 6):
  stats = {}
  stats['mean'] = df_col.mean()
  stats['minimum'] = min(df_col)
  stats['maximum'] = max(df_col)
  stats['variance'] = df_col.var()
  stats['mean windows'] = [0]*window_frac
  stats['variance windows'] = [0]*window_frac

  window = int(len(df_col)*(1/window_frac))

  beg, end = 0, window
  for i in range(window_frac):
    stats['mean windows'][i] = df_col[beg:end].mean()
    stats['variance windows'][i] = df_col[beg:end].var()
    beg = beg + window
    end = end + window
  return stats # Returns a dictionary

# Calculate statistics for every column in a dataframe
def get_df_stats(df, window_frac = 6):

  all_stats = {}
  for i, col in enumerate(df.columns):
    all_stats[f'Stats_{i}'] = get_col_stats(df[col], window_frac = window_frac)

  return all_stats # Returns a dictionary of dictionaries

# Prints the statistics from a one dimensional array stored in a dictionary
def print_col_stats(stats):
  significant_digits = 4
  print("Mean Value: ", round(stats['mean'], significant_digits))
  print("Variance Value: ", round(stats['variance'], significant_digits))
  print("Min & Max Value: ", round(stats['minimum'], significant_digits), round(stats['maximum'], significant_digits))
  print("Mean Windows (" + str(len(stats['mean windows'])) + "): ", [round(mean, significant_digits) for mean in stats['mean windows']])
  print("Variance Windows (" + str(len(stats['variance windows'])) +"): ", [round(var, significant_digits) for var in stats['variance windows']])

# Print the statistics from a dataframe stored in a dictionary of dictionaries
def print_df_stats(df, window_frac = 6):
  df_stats = get_df_stats(df, window_frac = window_frac)
  print("*************************************"*2)
  for col in df_stats:
    print(f"Column -{col}- Information:")
    print_col_stats(df_stats[col])
    print("*************************************"*2)
