import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def plot_line(x, y, k, b, line_label='Fitted line', color='r'):
    plt.plot(x, y, 'o', label='Original data', markersize=10)
    plt.plot(x, k * x + b, color, label=line_label)
    plt.legend()
    plt.show()


def generate_data():
    x = np.random.rand(100, 1)
    y = 2 + 3 * x + np.random.rand(100, 1)
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
    ani = FuncAnimation(fig, update, frames=len(k_values), init_func=init, blit=True, repeat=False,
                        interval=50)
    
    # Add legend and show plot
    plt.legend()
    plt.show()
