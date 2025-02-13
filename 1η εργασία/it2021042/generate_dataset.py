# Helper function for 1st project of Machine Learning course.
# Department of Informatics and Telematics
# Harokopio University of Athens
# 2024-2025

import numpy as np
import matplotlib.pyplot as plt


def generate_binary_problem(centers: np.ndarray, N: int = 100, p: int = 2) -> (np.ndarray, np.ndarray):
    """
    Generate a set of 2D points belonging in two classes

    N: int. Number of samples per class
    p: int. Number of dimensions
    centers: numpy.ndarray. A matrix whose columns correspond to the center
             of each class. Unit covariance matrix is assumed for all classes
    """
    rng = np.random.default_rng()
    # Class 0
    X0 = rng.multivariate_normal(centers[:, 0], np.eye(2), N)
    y0 = np.zeros(N)
    # Class 1
    X1 = rng.multivariate_normal(centers[:, 1], np.eye(2), N)
    y1 = np.ones(N)
    X = np.concatenate((X0, X1), axis=0)
    y = np.concatenate((y0, y1), axis=0)
    return (X, y)


def plot_binary_problem(X: np.ndarray, y: np.ndarray) -> None:
    """
    Plot a binary problem. This function assumes 2-D problem
    (just plots the first two dimensions of the data)
    """
    idx0 = (y == 0)
    idx1 = (y == 1)
    X0 = X[idx0, :2]
    X1 = X[idx1, :2]
    plt.plot(X0[:, 0], X0[:, 1], 'gx')
    plt.plot(X1[:, 0], X1[:, 1], 'ro')
    plt.show()


def plot_problem_and_line(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.float64) -> None:
    """
    Plot a binary problem and a line. This function assumes 2-D problem
    (just plots the first two dimensions of the data)
    """
    idx0 = (y == 0)
    idx1 = (y == 1)
    X0 = X[idx0, :2]
    X1 = X[idx1, :2]
    plt.plot(X0[:, 0], X0[:, 1], 'gx')
    plt.plot(X1[:, 0], X1[:, 1], 'ro')
    min_x = np.min(X, axis=0)
    max_x = np.max(X, axis=0)
    xline = np.arange(min_x[0], max_x[0], (max_x[0] - min_x[0]) / 100)
    yline = (w[0]*xline + b) / (-w[1])
    plt.plot(xline, yline, 'b')
    plt.show()


if __name__ == '__main__':
    print("Use with import statement.")
