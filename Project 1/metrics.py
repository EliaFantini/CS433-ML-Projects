import numpy as np


def accuracy_score(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    Computes the fraction of correctly classified points

    :param target: array of shape (n, ) with two possible values {-1, 1}
    :param prediction: array of shape (n, ) with two possible values {-1, 1}

    :return:
    """

    num_correct_objects = np.sum(target == prediction)

    return num_correct_objects / target.shape[0]


def precision_score(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    Computes the precision score tp / (tp + fp)

    :param target: array of shape (n, ) with two possible values {-1, 1}
    :param prediction: array of shape (n, ) with two possible values {-1, 1}

    :return:
    """

    tp = np.sum((prediction == 1) & (target == 1))
    fp = np.sum((prediction == 1) & (target == 0))

    return tp / (tp + fp)


def recall_score(target, prediction) -> float:
    """
    Computes the recall metric: tp / (tp + fn)

    :param target: array of shape (n, ) with two possible values {-1, 1}
    :param prediction: array of shape (n, ) with two possible values {-1, 1}

    :return:
    """

    tp = np.sum((prediction == 1) & (target == 1))
    fn = np.sum((prediction == 0) & (target == 1))

    return tp / (tp + fn)


def F1_score(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    Computes the following combination of two metrics (2 * precision * recall) / (precision + recall)

    :param target: array of shape (n, ) with two possible values {-1, 1}
    :param prediction: array of shape (n, ) with two possible values {-1, 1}

    :return:
    """

    precision = precision_score(target, prediction)
    recall = recall_score(target, prediction)

    f1 = (2 * precision * recall) / (precision + recall)

    return f1
