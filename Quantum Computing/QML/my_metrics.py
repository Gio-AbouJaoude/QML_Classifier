
from pandas import DataFrame
from my_qml import predict_all
from sklearn.metrics import confusion_matrix, classification_report

def metrics_test(weights, X_test, Y_test, n_circuits):
  # Calculate and return various classification metrics for the model's predictions on the test set

  labels = [i for i in range(n_circuits)]
  Y_estimate = predict_all(X_test, weights, n_circuits)
  metrics = classification_report(Y_test, Y_estimate, labels=labels, output_dict=True, zero_division=0)

  # Rename 'accuracy' key to 'model_accuracy'
  metrics['model_accuracy'] = metrics.pop('accuracy')
  metrics['ConfMatrix'] = confusion_matrix(Y_test, Y_estimate)

  # Extract and format precision, recall, and f1-score for each target class and model averages
  for target in range(n_circuits):
    target_metric_dic = metrics.pop(f"{target}")
    for metric_name in ['precision', 'recall', 'f1-score']:
      metrics[f"C_{target}_{metric_name}"] = target_metric_dic[metric_name]
  for model_metric in ['macro avg', 'weighted avg']:
    model_metric_dic = metrics.pop(model_metric)
    for metric_name in ['precision', 'recall', 'f1-score']:
      metrics[f"{model_metric}_{metric_name}"] = model_metric_dic[metric_name]
  return metrics

def percent(value):
  # Convert a decimal value to a percentage string representation
  return str(round(value*100, 2)) +'%'

def print_metrics_test(epoch, metrics, time_diff = None):
  # Print the metrics dictionary along with additional information

  accuracy = percent(metrics['model_accuracy'])
  f1 = percent(metrics['weighted avg_f1-score'])
  recall = percent(metrics['weighted avg_recall'])
  precision = percent(metrics['weighted avg_precision'])
  
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


def get_col_stats(df_col, window_frac = 6):
  # Calculate and return various statistics for a one-dimensional dataframe column
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

def get_df_stats(df, window_frac = 6):
  # Calculate and return various statistics for each column in a dataframe
  all_stats = {}
  for i, col in enumerate(df.columns):
    all_stats[f'Stats_{i}'] = get_col_stats(df[col], window_frac = window_frac)

  return all_stats # Returns a dictionary of dictionaries

def print_col_stats(stats):
  # Print the statistics from a one-dimensional array stored in a dictionary
  significant_digits = 4
  print("Mean Value: ", round(stats['mean'], significant_digits))
  print("Variance Value: ", round(stats['variance'], significant_digits))
  print("Min & Max Value: ", round(stats['minimum'], significant_digits), round(stats['maximum'], significant_digits))
  print("Mean Windows (" + str(len(stats['mean windows'])) + "): ", [round(mean, significant_digits) for mean in stats['mean windows']])
  print("Variance Windows (" + str(len(stats['variance windows'])) +"): ", [round(var, significant_digits) for var in stats['variance windows']])

def print_df_stats(df, window_frac = 6):
  # Print the statistics from a dataframe stored in a dictionary of dictionaries
  df_stats = get_df_stats(df, window_frac = window_frac)
  print("*************************************"*2)
  for col in df_stats:
    print(f"Column -{col}- Information:")
    print_col_stats(df_stats[col])
    print("*************************************"*2)
