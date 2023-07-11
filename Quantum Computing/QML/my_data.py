import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Constants for one-hot encoding and min-max scaling
one_hot_val = np.pi/2
min_max_scaling_val = np.pi

def difference(list_a, list_b):
  # Return the elements in list_a that are not in list_b
  return list(np.setdiff1d(list_a, list_b))

def col_num(df, col_name):
  # Return the column index of col_name in the dataframe df
  return df.columns.get_loc(col_name)

def one_hot(df, col_name):
  # Perform one-hot encoding on the column col_name of the dataframe df
  return pd.get_dummies(df[col_name]) 

def encode_col(df, col_name):
  # Encode the column col_name of the dataframe df as categorical codes
  return df[col_name].astype('category').cat.codes

def scale_data(df):
  # Scale the values of each column in the dataframe df to the range [0, 1]
  return (df - df.min()) / (df.max() - df.min())

def cat_columns(df, col_name_list):
  # Perform categorical encoding on the columns in col_name_list of the dataframe df
  non_cat_col_name_list = difference(df.columns, col_name_list)

  for col_name in col_name_list:
    if len(df[col_name].unique()) <= 2:
      # Perform categorical encoding on the columns in col_name_list of the dataframe df
      df[col_name] = encode_col(df, col_name)
    else:
      # If the column has more than 2 unique values, perform one-hot encoding
      df = pd.concat([df, one_hot(df, col_name)], axis = 1)
      df = df.drop([col_name], axis = 1)

  return df, difference(df.columns, non_cat_col_name_list)

def re_order(df, col_name_list):
  # Reorder the columns of the dataframe df based on the order in col_name_list
  return df[col_name_list]

def conv_np_to_df(np_array):
  # Convert a numpy array to a dataframe with column names as string representations of column indices
  return pd.DataFrame(np.array(np_array), columns = [str(i) for i in range(np_array.shape[1])])

def q_scale_data(df, cat_col_name_list = None):
  # Perform min-max scaling on the dataframe df, optionally handling categorical columns specified in cat_col_name_list
  
  # Convert df to a dataframe if it is not already of type pd.DataFrame
  df = conv_np_to_df(df) if not isinstance(df, pd.DataFrame) else df

  if cat_col_name_list:
    # Perform categorical encoding on the specified columns
    df, cat_col_name_list = cat_columns(df, cat_col_name_list)

  df =  min_max_scaling_val*scale_data(df)

  if cat_col_name_list:
    # Scale the categorical columns using a different scale factor
    df[cat_col_name_list] = (one_hot_val/min_max_scaling_val)*df[cat_col_name_list]

  return df

def target_data(df, col_name):
  # Split the dataframe df into features and target column based on col_name
  return df.drop(columns=[col_name]), encode_col(df, col_name)

def split_data(X, Y, test_size = 0.15, random_state = None):
  # Split the features (X) and target column (Y) into training and testing sets
  return tuple(train_test_split(np.array(X), np.array(Y), test_size = test_size, random_state = random_state))