import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def func(X, a, b):
    return a * ( x ** 3 ) + b

def func_exp_F(x, a, b):
    return a * np.exp(b * x) - a

if __name__ == "__main__":
    x = np.arange(-10, 10, 0.1)
    y =  func_exp_F(x, 0.1, 2) # GT

    noise = 5*1e5 * np.random.normal(size=x.size)
    # y = y + noise

    plt.plot(x, y)

    # popt, pcov = curve_fit(func_exp_F, x, y, method='lm')    
    # print(popt)
    # yp = func_exp_F(x, popt[0], popt[1])
    # plt.plot(x, yp)

    plt.show()