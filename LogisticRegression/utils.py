import os
import pickle
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np


'''
             name     role         type demographic                  description  units missing_values
0             Sex  Feature  Categorical        None         M, F, and I (infant)   None             no
1          Length  Feature   Continuous        None    Longest shell measurement     mm             no
2        Diameter  Feature   Continuous        None      perpendicular to length     mm             no
3          Height  Feature   Continuous        None           with meat in shell     mm             no
4    Whole_weight  Feature   Continuous        None                whole abalone  grams             no
5  Shucked_weight  Feature   Continuous        None               weight of meat  grams             no
6  Viscera_weight  Feature   Continuous        None  gut weight (after bleeding)  grams             no
7    Shell_weight  Feature   Continuous        None            after being dried  grams             no
8           Rings   Target      Integer        None  +1.5 gives the age in years   None             no
'''

def load_dataset(id=1, name="abalone"):
    # fetch dataset
    # test if file exists
    if not os.path.isfile(f"{name}.pkl"):
        dataset = fetch_ucirepo(id=id)
        pickle.dump(
            dataset,
            open(f"{name}.pkl", "wb"),
        )
    dataset = pickle.load(open(f"{name}.pkl", "rb"))

    # data (as pandas dataframes)
    X = dataset.data.features
    y = dataset.data.targets

    # metadata
    # print(dataset.metadata)

    # variable information
    # print(dataset.variables)
    return X, y

def one_hot_encode(df, column_name):
    one_hot_encoded = pd.get_dummies(df[column_name], prefix=column_name, )
    df = pd.concat([df, one_hot_encoded], axis=1)
    df = df.drop(columns=[column_name])
    return df

def normalize_dataset(df):
    return (df - df.mean()) / df.std()

def train_test_split(X, y, test_size=0.2):
    train_size = 1 - test_size

    shuffled_indices = np.random.permutation(len(X))
    X = X.iloc[shuffled_indices]
    y = y.iloc[shuffled_indices]

    train_X = X.iloc[:int(train_size * len(X))]
    test_X = X.iloc[int(train_size * len(X)):]

    train_y = y.iloc[:int(train_size * len(y))]
    test_y = y.iloc[int(train_size * len(y)):]

    return train_X, test_X, train_y, test_y

def prepare_dataset(df, do_one_hot_encode=False, column_names=[]):
    """
    one hot encode
    normalize
    """
    if do_one_hot_encode:
        for column_name in column_names:
            df = one_hot_encode(df, column_name)
    df = normalize_dataset(df)
    return df
