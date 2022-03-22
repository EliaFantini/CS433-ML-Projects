import numpy as np
import csv

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def train_test_split(data, y, test_size=0.3, random_state=23):

    """Gets train, test separation.

    Parameters
    ----------
    data : np.array
        numpy array of shape (n, m)
        n - number of objects
        m - number of features
    y: np.array
        numpy array of shape (n, ) of labels
    test_size : float from 0 to 1, optional (default=0.3)
        Fraction of the data to be test part.
    random_state : int or None, optional (default=23)
        Random state. If None then no shuffling.

    Returns
    -------
    arrays with train and test data and targets

    """
    dataset = np.concatenate([data, y[:, None]], axis=1)
    if random_state is not None:
        np.random.seed(random_state)
        np.random.shuffle(dataset)
    sep = int(data.shape[0] * (1 - test_size))

    X_train = dataset[:sep, :-1]
    y_train = dataset[:sep, -1]
    X_test = dataset[sep:, :-1]
    y_test = dataset[sep:, -1]

    return X_train, y_train, X_test, y_test


def predict_labels(data, weights):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
