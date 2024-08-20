import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pickle
import os
import pandas as pd
from ..PCA.PCA import myPCA

def plot_line(x, y, k, b, line_label='Fitted line', color='r'):
    plt.plot(x, y, 'o', label='Original data', markersize=10)
    plt.plot(x, k * x + b, color, label=line_label)
    plt.legend()
    plt.show()


def generate_data(dims = 1, number_of_points = 100):
    x = np.random.rand(number_of_points, dims)
    y = 2 + 3 * x + np.random.rand(number_of_points, 1)
    return x, y

def plot_data(x, y, label="Original data"):
    plt.plot(x, y, 'o', label=label, markersize=10)
    plt.legend()
    plt.show()


def plot_animation(x, y, k_values, b_values):
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    
    # Initialize a point plot and a line plot
    points, = ax.plot([], [], 'bo', label='Points')
    line, = ax.plot([], [], 'r-', label='Line')
    
    # Function to initialize the plot
    def init():
        points.set_data([], [])
        line.set_data([], [])
        return points, line
    
    # Update function for animation
    def update(frame):
        # Update points
        points.set_data(x, y)
        
        # Update line based on current slope and intercept
        k = k_values[frame]
        b = b_values[frame]
        x_line = np.linspace(min(x) - 1, max(x) + 1, 100)
        y_line = k * x_line + b
        line.set_data(x_line, y_line)
        
        return points, line
    
    # Create the animation
    FuncAnimation(fig, update, frames=len(k_values), init_func=init, blit=True, repeat=False,
                        interval=50)
    
    # Add legend and show plot
    plt.legend()
    plt.show()



"""
                  name     role         type                                        description   units
0              holiday  Feature  Categorical  US National holidays plus regional holiday, Mi...    None
1                 temp  Feature   Continuous                             Average temp in kelvin  Kelvin
2              rain_1h  Feature   Continuous     Amount in mm of rain that occurred in the hour      mm
3              snow_1h  Feature   Continuous     Amount in mm of snow that occurred in the hour      mm
4           clouds_all  Feature      Integer                          Percentage of cloud cover       %
5         weather_main  Feature  Categorical   Short textual description of the current weather    None
6  weather_description  Feature  Categorical  Longer textual description of the current weather    None
7            date_time  Feature         Date       Hour of the data collected in local CST time    None
8       traffic_volume   Target      Integer  Hourly I-94 ATR 301 reported westbound traffic...    None
(48204, 8) (48204, 1)
"""
def load_dataset():
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    # test if file exists
    if not os.path.isfile("metro_interstate_traffic_volume.pkl"):
        metro_interstate_traffic_volume = fetch_ucirepo(id=492)
        pickle.dump(
            metro_interstate_traffic_volume,
            open("metro_interstate_traffic_volume.pkl", "wb"),
        )
    metro_interstate_traffic_volume = pickle.load(
        open("metro_interstate_traffic_volume.pkl", "rb")
    )

    # data (as pandas dataframes)
    X = metro_interstate_traffic_volume.data.features
    y = metro_interstate_traffic_volume.data.targets

    # metadata
    # print(metro_interstate_traffic_volume.metadata)

    # variable information
    # print(metro_interstate_traffic_volume.variables)
    return X, y

def one_hot_encode(df, column_name):
    one_hot_encoded = pd.get_dummies(df[column_name], prefix=column_name, )
    df = pd.concat([df, one_hot_encoded], axis=1)
    df = df.drop(columns=[column_name])
    return df

def to_date_time(df, column_name='date_time'):
    df[column_name] = pd.to_datetime(df[column_name])
    df['year'] = df[column_name].dt.year
    df['month'] = df[column_name].dt.month
    df['day'] = df[column_name].dt.day
    df['hour'] = df[column_name].dt.hour
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

def prepare_dataset(df):
    """
    one hot encode
    date time to year, month, day, hour
    normalize
    """
    df = one_hot_encode(df, 'holiday')
    df = one_hot_encode(df, 'weather_main')
    df = one_hot_encode(df, 'weather_description')
    df = to_date_time(df)
    df = normalize_dataset(df)
    df = pd.DataFrame(myPCA(df.to_numpy(), percentage=0.95))
    return df
