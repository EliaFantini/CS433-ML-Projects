from typing import Tuple

import numpy as np


def compute_mse_gradient(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of MSE

    :param y: Target shape (n, )
    :param x: Array of observations shape (n, m)
    :param w: Weights

    :returns gradient of the MSE
    """
    e = y - x @ w
    return (-1 / x.shape[0]) * x.T @ e


def compute_mse_loss(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> float:
    """
    Calculate the MSE loss
    :param y: Target shape (n, )
    :param x: Array of observations shape (n, m)
    :param w: Weights

    :returns scalar loss of the function
    """
    e = y - x @ w
    loss = (1 / (2 * y.shape[0])) * e.T @ e

    return loss


def sigmoid(tx: np.ndarray) -> np.ndarray:
    """
    Our implementation of a classic sigmoid function

    :param tx: feature values of shape (n, m)

    :return sigmoid(x)
    """

    exp_x = np.exp(tx)
    return exp_x / (1 + exp_x)


def predict_probabilities(tx: np.array, w: np.array) -> np.array:
    """
     Function that makes class predictions in range [0, 1]

     :param tx: feature values of shape (n, m)
     :param w: feature weights of shape (m, )

     where n - number of data points, m - number of features

     :return: probabilities of data points being of class 1, shape (n, )
     """

    predicted_prob = sigmoid(tx @ w)
    return predicted_prob


def logistic_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """
    Computes the negative log likelihood loss

    :param y: target labels of shape (n, )
    :param tx: training data of shape (n, m)
    :param w:

    :return:
    """
    pred = tx @ w
    return np.mean(np.log(1 + np.exp(pred)) - y * pred)


def calculate_gradient_logistic_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Calculates the gradient of a logistic loss

    :param y: target labels of shape (n, )
    :param tx: training data of shape (n, m)
    :param w: feature weights of shape (m, )

    where n - number of data points, m - number of features

    :return:
    """

    gradient = np.mean(tx * (predict_probabilities(tx, w) - y)[:, np.newaxis], axis=0)
    return gradient


# -------------- Models --------------


def least_squares(y: np.ndarray, tx: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Analytical solution of the least squares problem

    :param y: target labels of shape (n, )
    :param tx: feature values of shape (n, m)

    :return:
    """
    w = np.linalg.solve((tx.T @ tx), tx.T @ y)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_: float) -> Tuple[np.ndarray, float]:
    """
    Implementation of a ridge regression, which finds analytical solution to the problem

    :param y: target labels of shape (n, )
    :param tx: feature values of shape (n, m)
    :param lambda_: regularization parameter
    :return:
    """

    num_objects = tx.shape[0]
    num_features = tx.shape[1]
    a = tx.T @ tx + 2 * num_objects * lambda_ * np.identity(num_features)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def least_squares_GD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                     max_iters: int, gamma: float) -> Tuple[np.ndarray, list]:
    """
    Gradient Descent algorithm for linear regression
    with Mean Squared Error loss function

    :param y: target labels of shape (n, )
    :param tx: training data of shape (n, m)
    :param initial_w: initial weight estimation of shape (m, )
    :param max_iters: maximum iterations allowed
    :param gamma: learning rate

    :return: tuple with the first element being the weights of the model of shape (m, )
                    and the second element being the final loss
    """
    losses = []
    w = initial_w
    for _ in range(max_iters):
        grad = compute_mse_gradient(y, tx, w)
        w = w - np.dot(gamma, grad)
        loss = compute_mse_loss(y, tx, w)
        losses.append(loss)
    return w, losses


def least_squares_SGD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                      max_iters: int, gamma: float) -> Tuple[np.ndarray, float]:
    """
    Stochastic Gradient Descent algorithm for linear regression
    with Mean Squared Error loss function

    :param y: target labels of shape (n, )
    :param tx: training data of shape (n, m)
    :param initial_w: initial weight estimation of shape (m, )
    :param max_iters: maximum iterations allowed
    :param gamma: learning rate

    :return: tuple with the first element being the weights of the model of shape (m, )
                    and the second element being the final loss
    """

    w = initial_w
    for _ in range(max_iters):
        index = np.random.randint(0, y.shape[0], size=1)
        y_n = y[index]
        x_n = tx[index]
        stoch_grad = compute_mse_gradient(y_n, x_n, w)
        w = w - gamma * stoch_grad
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def least_squares_batch_GD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                           max_iters: int, gamma: float, batch_size: int) -> Tuple[np.ndarray, list]:
    """
    Stochastic Gradient Descent algorithm for linear regression
    with Mean Squared Error loss function

    :param y: target labels of shape (n, )
    :param tx: training data of shape (n, m)
    :param initial_w: initial weight estimation of shape (m, )
    :param max_iters: maximum iterations allowed
    :param gamma: learning rate
    :param batch_size: batch size for gradient descent to take on each iteration

    :return: tuple with the first element being the weights of the model of shape (m, )
                    and the second element being a list of losses during iterations
    """
    w = initial_w
    data_n = y.shape[0]
    indices = range(data_n)
    losses = []

    for _ in range(max_iters):
        batch_indices = np.random.choice(indices, size=batch_size, replace=False)
        y_n = y[batch_indices]
        x_n = tx[batch_indices]
        batch_grad = compute_mse_gradient(y_n, x_n, w)
        w = w - gamma * batch_grad
        loss = compute_mse_loss(y, tx, w)
        losses.append(loss)

    return w, losses


def logistic_regression(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                        max_iters: int, gamma: float) -> Tuple[np.ndarray, float]:
    """
    Function for training a logistic regression

    :param y: target labels of shape (n, )
    :param tx: training data of shape (n, m)
    :param initial_w: initial weight estimation of shape (m, )
    :param max_iters: maximum iterations allowed
    :param gamma: learning rate

    where n - number of data points, m - number of features

    :return: tuple with the first element being the weights of the model of shape (m, )
                    and the second element being the final loss
    """

    w = initial_w
    for _ in range(max_iters):
        grad = calculate_gradient_logistic_loss(y, tx, w)
        w = w - gamma * grad
    loss = logistic_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                            max_iters: int, gamma: float, lambda_: float) -> Tuple[np.ndarray, float]:
    """
    Function for training a logistic regression

    :param y: target labels of shape (n, )
    :param tx: training data of shape (n, m)
    :param initial_w: initial weight estimation of shape (m, )
    :param max_iters: maximum iterations allowed
    :param gamma: learning rate
    :param lambda_: regularization parameter

    where n - number of data points, m - number of features

    :return: tuple with the first element being the weights of the model of shape (m, )
                and the second element being the final loss
    """

    w = initial_w
    for i in range(max_iters):
        gradient = calculate_gradient_logistic_loss(y, tx, w) + lambda_ * w
        w = w - gradient * gamma
    loss = logistic_loss(y, tx, w) + (lambda_ / 2) * np.sum(w ** 2)
    return w, loss
