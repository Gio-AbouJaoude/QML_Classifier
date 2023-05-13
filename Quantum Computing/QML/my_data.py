import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

one_hot_val = np.pi/2
min_max_scaling_val = np.pi

def difference(list_a, list_b):
  return list(np.setdiff1d(list_a, list_b))

def col_num(df, col_name):
  return df.columns.get_loc(col_name)

def one_hot(df, col_name):
  return pd.get_dummies(df[col_name]) 

def encode_col(df, col_name):
  return df[col_name].astype('category').cat.codes

def scale_data(df):
  return (df - df.min()) / (df.max() - df.min())

def cat_columns(df, col_name_list):
  non_cat_col_name_list = difference(df.columns, col_name_list)

  for col_name in col_name_list:
    if len(df[col_name].unique()) <= 2:
      df[col_name] = encode_col(df, col_name)
    else:
      df = pd.concat([df, one_hot(df, col_name)], axis = 1)
      df = df.drop([col_name], axis = 1)

  return df, difference(df.columns, non_cat_col_name_list)

def re_order(df, col_name_list):
  return df[col_name_list]

def conv_np_to_df(np_array):
  return pd.DataFrame(np.array(np_array), columns = [str(i) for i in range(np_array.shape[1])])

def q_scale_data(df, cat_col_name_list = None):

  df = conv_np_to_df(df) if not isinstance(df, pd.DataFrame) else df

  if cat_col_name_list:
    df, cat_col_name_list = cat_columns(df, cat_col_name_list)

  df =  min_max_scaling_val*scale_data(df)

  if cat_col_name_list:
    df[cat_col_name_list] = (one_hot_val/min_max_scaling_val)*df[cat_col_name_list]

  return df

def target_data(df, col_name):
  return df.drop(columns=[col_name]), encode_col(df, col_name)

def split_data(X, Y, test_size = 0.15, random_state = None):
  return tuple(train_test_split(np.array(X), np.array(Y), test_size = test_size, random_state = random_state))