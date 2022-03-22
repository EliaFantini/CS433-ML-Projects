import numpy as np
from tqdm import tqdm

__COLUMNS_W_MISSING_VALUES = [0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28]
__COLUMNS_TO_LOG = [0, 1, 2, 3, 5, 7, 8, 9, 10, 13, 16, 19, 21, 22, 25, 28]


def standardize(x):
    """Standardize the original data set."""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / std


def add_bias(x):
    """Adds bias term to features"""
    tx = np.c_[np.ones(x.shape[0]), x]
    return tx


def get_corr_matrix(X):
    n = X.shape[1]
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = np.corrcoef(X[:, i], X[:, j])[0, 1]
    return corr


def drop_corr_features(X, threshold=0.9):
    """
    Reduce highly correlated features
    Parameters
    --------------
    X: np.array
        Observations (shape nxm)
    threshold: float from 0 to 1
        The threshold for reduction
    Returns
    -----------------
    X_selected: np.array
        Observations with correlation < threshold (shape nxk, where k<=m)

    selected_columns: inds of selected columns
    """
    corr = get_corr_matrix(X)
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr[i, j] >= threshold:
                if columns[j]:
                    columns[j] = False
    selected_columns = np.arange(X.shape[1])[columns]
    return X[:, columns], selected_columns


def find_missing_values(X):
    mask = X == -999
    mask = np.sum(mask, axis=0)
    return np.where(mask > 0)[0]


def onehot_encoding(data,  feature_column_id, value):
    onehot = np.zeros((data.shape[0], value))
    feature = data[:, feature_column_id]
    for i in range(value):
        onehot[:, i] = (feature == i)
    return onehot


def onehot_reduced(data, columns_ids):
    new_data = data.copy()
    onehot = np.zeros((data.shape[0], len(columns_ids)))
    for i, j in enumerate(columns_ids):
        onehot[:, i] = data[:, j] == -999
        ##?
        new_data[:, j][new_data[:, j] == -999] = new_data[:, j][new_data[:, j] != -999].mean()
    return onehot, new_data


def log_features(X, columns_to_log=__COLUMNS_TO_LOG):
    X[:, columns_to_log] = X[:, columns_to_log] - np.min(X[:, columns_to_log], axis=0) + 0.1
    for i in columns_to_log:
        X[:, i] = np.log(X[:, i])
    return X


def min_max_scaler(X):
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))


def make_cross_features(inp):
    additional_features = inp.copy()
    for f1 in tqdm(range(30)):
        for f2 in range(f1 + 1):
            additional_features = np.hstack([additional_features, inp[:, [f1]] * inp[:, [f2]]])

    return additional_features


def handle_one_hot_encoding(data, feature_column_id=22, value=4, missing_value_columns=__COLUMNS_W_MISSING_VALUES):
    one_hot_matrix = onehot_encoding(data, feature_column_id, value)
    one_hot_reduced_matrix, new_data = onehot_reduced(data, missing_value_columns)
    new_data_dropped = new_data[:, np.arange(new_data.shape[1]) != feature_column_id]
    result = np.concatenate([new_data_dropped, one_hot_matrix, one_hot_reduced_matrix], axis=1)
    return result


def final_preprocessing(data, feature_column_id=22, value=4, missing_value_columns=__COLUMNS_W_MISSING_VALUES):
    processed_data = handle_one_hot_encoding(data,
                                             feature_column_id=feature_column_id,
                                             value=value, missing_value_columns=missing_value_columns)
    processed_data = log_features(processed_data)
    processed_data = standardize(processed_data)
    processed_data = drop_corr_features(processed_data)[0]
    processed_data = make_cross_features(processed_data)
    processed_data = add_bias(processed_data)
    return processed_data
