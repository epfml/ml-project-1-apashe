#imports
import numpy as np
from helpers import *
from implementations import *
from utils import *
import matplotlib.pyplot as plt

def main(TRAIN=0):
    # Load raw data
    data_path = 'dataset'
    x_tr_raw, x_te_raw, y_tr_raw, train_ids, test_ids = load_csv_data(data_path, sub_sample=False)

    # Remove irrelevant columns
    columns_to_remove = np.concatenate((range(1, 25), range(54, 56)))

    x_tr_remove_col = np.delete(x_tr_raw, columns_to_remove, axis=1)
    x_te_remove_col = np.delete(x_te_raw, columns_to_remove, axis=1)

    # Set the "don't know" and "refused" values to NaN
    x_tr_dk = replace_dk_values_with_nan(x_tr_remove_col)
    x_val_dk = replace_dk_values_with_nan(x_te_remove_col)

    # Set the optimal hyperparameters
    dup = 3
    deg = 5
    lambda_ = 0.0002805263157894737

    # Replace NaN values with the mean in each feature
    x_tr_no_nan, x_val_no_nan = nan_to_mean(x_tr_dk, x_val_dk)

    # Standardize the features
    x_tr_std, x_val_std = standardize(x_tr_no_nan, x_val_no_nan)

    # Duplicate rows with label 1 to balance dataset
    x_tr_duplicated, y_tr_duplicated = duplicate_1rows(x_tr_std, y_tr_raw, dup)

    # Create a polynomial basis of the inputs
    x_poly_tr = build_poly(x_tr_duplicated, deg)
    x_poly_val = build_poly(x_val_std, deg)

    # Add intercept
    tx = np.c_[np.ones((x_poly_tr.shape[0], 1)), x_poly_tr]

    # Train the model if TRAIN = 1, Load w and test if TRAIN = 0
    if TRAIN:   
        w, loss = ridge_regression(y_tr_duplicated, tx, lambda_)
        np.savetxt("w.txt", w)
    else:
        w = np.loadtxt("w.txt")
        y_pred = predict(x_poly_val, w, "gd")
        name = "testing run.py 3"
        create_csv_submission(test_ids, y_pred, name)

if __name__ == "__main__":
    TRAIN = 0
    main(TRAIN)