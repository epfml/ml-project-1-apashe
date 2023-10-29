"""Data wrangling Functions"""
import numpy as np
import math
# Just to test
from sklearn.metrics import f1_score


def evaluate(y_pred, y_gt):
    result = y_pred==y_gt
    print(f'Accuracy: {list(result).count(True)/len(result)}')
    print(f'F1-score: {f1_score(y_gt,y_pred)}')
    return f1_score(y_gt,y_pred)

def evaluate1(y_pred, y_gt):
    result = y_pred==y_gt
    return list(result).count(True)/len(result), f1_score(y_gt,y_pred)


def drop_column(arr, column_to_drop):
    """
    Drop a specified column from a NumPy array.

    Parameters:
        arr (numpy.ndarray): The input NumPy array.
        column_to_drop (int): The index of the column to drop (0-based index).

    Returns:
        numpy.ndarray: The modified array with the specified column removed.
    """

    return np.delete(arr, column_to_drop, axis=1)


def delete_outliers(x_train_st, y_train_st,Z_threshold=3):
    # Calculate Z-scores for each feature in X_train
    z_scores = np.abs(x_train_st)
    # Find the indices of outliers based on the Z-scores
    outlier_indices = np.any(z_scores > Z_threshold, axis=1)
    
    # Remove outliers from X_train and y_train
    X_clean = np.delete(x_train_st, outlier_indices, axis=0)
    y_clean = np.delete(y_train_st, outlier_indices, axis=0)
    
    return X_clean, y_clean

def standardize(x_train, x_test):
    std_train = np.nanstd(x_train, axis=0)
    non_zero_std_columns = np.where(std_train != 0)[0]

    x_train_st = (x_train[:, non_zero_std_columns] - np.nanmean(x_train[:, non_zero_std_columns], axis=0)) / std_train[non_zero_std_columns]
    x_test_st = (x_test[:, non_zero_std_columns] - np.nanmean(x_test[:, non_zero_std_columns], axis=0)) / std_train[non_zero_std_columns]

    return x_train_st, x_test_st


def standardize2(x_train,x_test):
    x_train_st = (x_train - np.nanmean(x_train,axis=0))/np.nanstd(x_train,axis=0)
    x_test_st = (x_test - np.nanmean(x_train,axis=0))/np.nanstd(x_train,axis=0)
    return x_train_st, x_test_st


def duplicate_1rows(x_train, y_train,n=4):
    # Find the indices of rows where y_tr equals 1
    indices_to_duplicate = np.where(y_train == 1)[0]

    # Duplicate the selected rows n times
    x_tr_duplicated = np.concatenate([x_train, x_train[indices_to_duplicate]], axis=0)
    y_tr_duplicated = np.concatenate([y_train, y_train[indices_to_duplicate]], axis=0)

    # Repeat n-1 more times to get a total of n duplicates
    for _ in range(n-1):
        x_tr_duplicated = np.concatenate([x_tr_duplicated, x_train[indices_to_duplicate]], axis=0)
        y_tr_duplicated = np.concatenate([y_tr_duplicated, y_train[indices_to_duplicate]], axis=0)
    return x_tr_duplicated, y_tr_duplicated 


def duplicate_1rows(x_train, y_train, n=4):
    # Find the indices of rows where y_train equals 1
    indices_to_duplicate = np.where(y_train == 1)[0]

    whole_part = math.floor(n)  # Whole number part of n
    fractional_part = n - whole_part  # Fractional part of n

    # Duplicate the selected rows whole_part times
    x_tr_duplicated = np.concatenate([x_train] + [x_train[indices_to_duplicate]] * whole_part, axis=0)
    y_tr_duplicated = np.concatenate([y_train] + [y_train[indices_to_duplicate]] * whole_part, axis=0)

    # Add floor(fractional_part * number_of_1_columns) more duplicates
    num_fractional_duplicates = math.floor(fractional_part * len(indices_to_duplicate))
    
    if num_fractional_duplicates > 0:
        x_tr_duplicated = np.concatenate([x_tr_duplicated] + [x_train[indices_to_duplicate][:num_fractional_duplicates]], axis=0)
        y_tr_duplicated = np.concatenate([y_tr_duplicated] + [y_train[indices_to_duplicate][:num_fractional_duplicates]], axis=0)

    return x_tr_duplicated, y_tr_duplicated

def filter_nan_threshold(x_train, x_test, threshold):
    if threshold == 0:
        nan_columns = np.any(np.isnan(x_train), axis=0)  # Check for NaN in each column
        x_train_filtered = x_train[:, ~nan_columns]  # ~nan_columns to remove NaN-containing columns
        x_test_filtered = x_test[:, ~nan_columns]
    else:
        nan_percentage = np.isnan(x_train).mean(axis=0)
        keep_columns = nan_percentage <= threshold
        x_train_filtered = x_train[:, keep_columns]
        x_test_filtered = x_test[:, keep_columns]
    return x_train_filtered, x_test_filtered

def nan_to_zero(x_train, x_test):
    x_train_nonan = np.nan_to_num(x_train, nan=0)
    x_test_nonan = np.nan_to_num(x_test, nan=0)
    return x_train_nonan, x_test_nonan

def nan_to_mean(x_train, x_test):
    # Calculate the mean of each column while ignoring NaN values
    column_means = np.nanmean(x_train, axis=0)
    
    # Replace NaN values in x_train and x_test with the respective column means
    x_train_nonan = np.where(np.isnan(x_train), column_means, x_train)
    x_test_nonan = np.where(np.isnan(x_test), column_means, x_test)
    
    return x_train_nonan, x_test_nonan

def replace_dk_values_with_nan(x_train):
    x_train = x_train.astype(float)  # Convert the matrix to a float type to allow NaN
    
    max_value = np.max(x_train, axis=0)
    
    for i in range(x_train.shape[1]):
        if max_value[i] == 9:
            x_train[(x_train[:, i] == 7) | (x_train[:, i] == 9), i] = np.nan
        elif max_value[i] == 99:
            x_train[(x_train[:, i] == 77) | (x_train[:, i] == 99), i] = np.nan
        elif max_value[i] == 999:
            x_train[(x_train[:, i] == 777) | (x_train[:, i] == 999), i] = np.nan
        elif max_value[i] == 9999: 
            x_train[(x_train[:, i] == 7777) | (x_train[:, i] == 9999), i] = np.nan    # ADDED 7777
        elif max_value[i] == 99900:
            x_train[(x_train[:, i] == 99900 ), i] = np.nan
        # following is new
        elif max_value[i] == 99999:
            x_train[(x_train[:, i] == 99999), i] = np.nan
        elif max_value[i] == 999999:
            x_train[(x_train[:, i] == 777777) | (x_train[:, i] == 999999), i] = np.nan
    return x_train


def predict_and_evaluate(x, w, y, regression_type="gd"):
    tx_te = np.c_[np.ones((x.shape[0], 1)), x]
    y_pred = tx_te @ w
    if regression_type=="gd":
        y_pred = [1 if y > 0 else -1 for y in y_pred]
    elif regression_type=="lr":
        y_pred = [1 if y > 0.5 else -1 for y in y_pred]
    f1 = evaluate(y_pred, y)
    return f1


def predict(x, w, regression_type="gd"):
    tx_te = np.c_[np.ones((x.shape[0], 1)), x]
    y_pred = tx_te @ w
    if regression_type=="gd":
        y_pred = [1 if y > 0 else -1 for y in y_pred]
    elif regression_type=="lr":
        y_pred = [1 if y > 0.5 else -1 for y in y_pred]
    return y_pred


def build_poly(x, degree):
    N, D =  x.shape
    poly = np.zeros((N, D*(degree)))

    for j in range(degree):
        poly[:, (D*j):((D*j)+D)] = x ** (j+1)

    return poly


def split_data(x, y, ratio, seed=42):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # ***************************************************
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Calculate the number of samples for training and testing
    num_samples = len(x)
    num_train_samples = int(np.floor(ratio * num_samples))
    
    # Create a shuffled index array
    shuffled_indices = np.random.permutation(num_samples)
    
    # Split the indices into training and testing sets
    train_indices = shuffled_indices[:num_train_samples]
    test_indices = shuffled_indices[num_train_samples:]
    
    # Use the indices to split the data and labels
    x_tr = x[train_indices]
    x_te = x[test_indices]
    y_tr = y[train_indices]
    y_te = y[test_indices]
    
    return np.array(x_tr), np.array(x_te), np.array(y_tr), np.array(y_te)