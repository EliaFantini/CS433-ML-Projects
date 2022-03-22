import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from implementations import (least_squares_GD, least_squares_SGD, least_squares_batch_GD,
                             ridge_regression, logistic_regression, reg_logistic_regression, predict_probabilities)
from metrics import accuracy_score, F1_score
from utils import predict_labels


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)

    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, tx, max_iters, k_indices, k, hyperparameter, regression_function, batch_size=100, lambda_=0.1):
    # get k'th subgroup in test, others in train
    train_tx = np.delete(tx, k_indices[k], 0)
    train_y = np.delete(y, k_indices[k], 0)
    w = np.zeros(tx.shape[1])

    # form data with polynomial degree
    # ridge regression
    if regression_function is least_squares_GD:
        w, _ = regression_function(train_y, train_tx, np.zeros(tx.shape[1]), max_iters, hyperparameter)
    if regression_function is least_squares_SGD:
        w, _ = least_squares_SGD(y, tx, np.zeros(tx.shape[1]), max_iters, hyperparameter)
    if regression_function is least_squares_batch_GD:
        w, _ = regression_function(train_y, train_tx, np.zeros(tx.shape[1]), max_iters, hyperparameter, batch_size)
    if regression_function is ridge_regression:
        w, _ = regression_function(train_y, train_tx, hyperparameter)
    if regression_function is logistic_regression:
        w, _ = regression_function(train_y, train_tx, np.zeros(tx.shape[1]), max_iters, hyperparameter)
    if regression_function is reg_logistic_regression:
        w, _ = regression_function(train_y, train_tx, np.zeros(tx.shape[1]), max_iters , hyperparameter, lambda_)

    return w


def tune_hyperparameter(y, tx, seed=23, max_iters=200, k_fold=5, hyperparameters=None, hyperparameters_2=[0],
                        regression_function=None , verbose= False):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    best_hyperparameter = 0
    best_hyperparameter_2 = 0
    best_score = 0
    best_w = np.zeros(tx.shape[1])
    iter = 1
    for hyperparameter in hyperparameters:
        for hyperparameter_2 in hyperparameters_2:
            w_sum = np.zeros(tx.shape[1])
            for i in range(k_fold):
                if regression_function is reg_logistic_regression:
                    w = cross_validation(y, tx, max_iters, k_indices, i, hyperparameter,
                                         regression_function, lambda_=hyperparameter_2)
                elif regression_function is least_squares_batch_GD:
                    w = cross_validation(y, tx, max_iters, k_indices, i, hyperparameter,
                                         regression_function, batch_size=hyperparameter_2)
                else:
                    w = cross_validation(y, tx, max_iters, k_indices, i, hyperparameter,
                                         regression_function)
                w_sum += w
            w = w_sum / k_fold
            if (regression_function is logistic_regression) or (regression_function is reg_logistic_regression) :
                prediction = predict_probabilities(tx, w) > 0.5
            else:
                prediction = predict_labels(weights=w, data=tx)
            accuracy = accuracy_score(y, prediction)
            if accuracy > best_score:
                best_score = accuracy
                best_hyperparameter = hyperparameter
                best_hyperparameter_2 = hyperparameter_2
                best_w = w
            if verbose:
                print(str(iter) + ": Parameter 1: " + str(hyperparameter) + " Parameter 2: " + str(hyperparameter_2) + " Accuracy: " + str(accuracy))
            iter += 1
    print("Best hyperparameters: hyperparameter 1 = " + str(best_hyperparameter) + " hyperparameter 2 = " + str(best_hyperparameter_2) + " accuracy: " + str(best_score))

    return best_hyperparameter, best_hyperparameter_2, best_w


def run_experiment(method,
                   X_train, y_train,
                   X_test, y_test,
                   loss_func, gamma, initial_w,
                   epochs=100, step=20,
                   verbose=False, lambda_=None):
    losses_train = []
    losses_test = []
    acc = []
    f1 = []
    for i in tqdm(range(epochs)):
        params = {}

        if lambda_:
            params['lambda_'] = lambda_

        w, loss = method(y_train, X_train, initial_w, step, gamma, **params)
        initial_w = w
        losses_train.append(loss)

        test_loss = loss_func(y_test, X_test, w)
        losses_test.append(test_loss)

        predicted_values = predict_probabilities(X_test, w) > 0.5

        accuracy = accuracy_score(target=y_test, prediction=predicted_values)
        acc.append(accuracy)

        f1_score = F1_score(target=y_test, prediction=predicted_values)
        f1.append(f1_score)

        if verbose:
            print(f'Epoch {(i+1)}, accuracy: {accuracy}')

    print(f'Accuracy max: {np.max(acc)}, F1 max: {np.max(f1)}')
    return losses_train, losses_test, acc, f1


def vizualisation(losses_train, losses_test, acc, f1):
    plt.style.use('ggplot')
    fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), sharex=True)

    ax1.plot(losses_train, label='train loss')
    ax1.plot(losses_test, label='test loss')
    ax2.plot(acc, label='accuracy')
    ax2.plot(f1, label='f1 score')
    ax1.set_xlabel('Epochs', fontsize=20)
    ax1.set_ylabel('Logistic loss', fontsize=20)
    ax2.set_xlabel('Epochs', fontsize=20)
    ax2.set_ylabel('Acc and f1 score', fontsize=20)
    ax1.legend(fontsize=20)
    ax2.legend(fontsize=20)
