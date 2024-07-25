# import linear regression form sklean
from sklearn.linear_model import LinearRegression
import numpy as np
from utils import plot_line, plot_data, generate_data


def sklearn_Linear_Regression(x, y):
    reg = LinearRegression()
    reg.fit(x, y)
    print(f'Coefficients: {reg.coef_}, Intercept: {reg.intercept_}')
    print('Variance score: ', reg.score(x, y))
    plot_line(x, y, reg.coef_, reg.intercept_, line_label='sklearn Fitted line')

def myLinearRegression(x, y):
    A = np.stack([x, np.ones((len(x), 1))]).T[0]
    b = y
    k, b = np.linalg.lstsq(A, b, rcond=None)[0]
    print(k, b)
    plot_line(x, y, k, b, line_label='My Fitted line', color='b')


if __name__ == '__main__':
    x, y = generate_data()
    plot_data(x, y)
    print("Sklearn Linear Regression:")
    sklearn_Linear_Regression(x, y)
    print("My Linear Regression:")
    myLinearRegression(x, y)
