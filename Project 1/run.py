import sys
sys.path.append('../')

import numpy as np

from os.path import join

from utils import load_csv_data, create_csv_submission
from implementations import logistic_regression, predict_probabilities
from preprocessing import final_preprocessing


DATA_PATH = 'data/'
y, X, _ = load_csv_data(join(DATA_PATH, 'train.csv'))

# preprocessing data
X_processed = final_preprocessing(X)

# Setting hyperparameters to selected values
gamma = 0.16681005372000587
max_iters = 2000
initial_w = np.zeros(X_processed.shape[1])
# Training the model
w, _ = logistic_regression(y == 1, X_processed, initial_w, max_iters, gamma)

# Loading and preprocessing the test (submission) data
X_test = np.genfromtxt(join(DATA_PATH, 'test.csv'), delimiter=",", skip_header=1)[:, 2:]
X_test_processed = final_preprocessing(X_test)
test_ids = np.arange(X_test_processed.shape[0]) + 350000

# Making labels out of our probabilistic predictions
prediction = predict_probabilities(tx=X_test_processed, w=w) > 0.5
predicted_labels = [1 if i == 1 else -1 for i in prediction]

# creating final csv file with our predictions
create_csv_submission(test_ids, predicted_labels, 'submission.csv')
