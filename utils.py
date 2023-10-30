"""Useful functions"""
import numpy as np

## UTILS USED IN THE 6 ML METHOD

# Set a random seed for reproducibility
np.random.seed(1)

def compute_mse(y, tx, w):
    """compute the loss by mse
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.

    Returns:
        mse: scalar corresponding to the mse with factor (1 / 2 n) in front of the sum
    """
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_gradient_mse(y, tx, w):
    """Computes the gradient at w for linear regression.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - np.matmul(tx, w)
    grad = (-1/y.shape[0])*np.matmul(np.transpose(tx), e)
    return grad


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return (1/(1+np.exp(-t)))


def compute_logistic_loss(y, tx, w):
    """compute the logistic loss.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )

    Returns:
        a non-negative loss
    """

    return -np.mean(y*np.log(sigmoid(tx@w))+(1-y)*np.log(1-sigmoid(tx@w)))



def compute_gradient_logistic_loss(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """

    N = len(y)
    return (1/N)*(tx.T)@((sigmoid(tx@w))-(y))


def gradient_descent_for_logistic_regression(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    gradient = compute_gradient_logistic_loss(y, tx, w)
    w_next = w - gamma * gradient
    
    loss = compute_logistic_loss(y, tx, w_next)

    return loss, w_next



def gradient_descent_for_penalized_logistic_regression(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """

    gradient_logistic = compute_gradient_logistic_loss(y, tx, w)
    gradient_regularization = lambda_ * w

    gradient = gradient_logistic + 2 * gradient_regularization

    w = w - gamma * gradient
    


    loss_logistic = compute_logistic_loss(y, tx, w)
    loss_regularization = 0.5 * lambda_ * np.sum(w**2)
    loss = loss_logistic + 2 * loss_regularization

    return loss, w



def compute_stochastic_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    e = y - tx@w
    N = y.size
    gradient = -(1/N)*(tx.T @ e)
    return gradient


def stochastic_gradient_descent_for_mse(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        stoch_gradient = [0,0]
        stoch_loss = 0
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = compute_stochastic_gradient(minibatch_y,minibatch_tx,w)
            loss = compute_logistic_loss(minibatch_y, minibatch_tx, w)
            loss = compute_mse(minibatch_y, minibatch_tx, w)
            # loss = compute_loss(minibatch_y, minibatch_tx, w)
            stoch_gradient += grad
            stoch_loss += loss
        w = w - gamma*stoch_gradient

        ws.append(w)
        losses.append(stoch_loss)

        print(
            "SGD iter. {bi}/{ti}: loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss
            )
        )
    return losses, ws



## GENERAL UTILS FOR DATA WRANGLING

def compute_f1(y_true, y_pred):
    """
    Calculate the F1 score using NumPy.

    Parameters:
    - y_true: NumPy array or list, true labels
    - y_pred: NumPy array or list, predicted labels

    Returns:
    - F1 score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate true positives, false positives, and false negatives
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == -1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == -1))

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    # Calculate the F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score


def predict(x, w, regression_type="mse"):
    """
    Make predictions using a linear model with the given weights.

    Parameters:
    x (array-like): Feature data for prediction.
    w (array-like): Weight parameters for the linear model.
    regression_type (str, optional): Type of regression ('mse' for mean squared error, 'lr' for logistic regression). Default is 'mse'.

    Returns:
    array-like: Predicted labels based on the input data and regression type.

    This function takes feature data and a set of weight parameters for a linear model and
    produces predictions based on the specified regression type. For mean squared error loss ('mse'),
    the prediction is 1 if the output is greater than 0, otherwise -1. For logistic regression ('lr'),
    the prediction is 1 if the output is greater than 0.5, otherwise -1.
    """
    tx_te = np.c_[np.ones((x.shape[0], 1)), x]
    y_pred = tx_te @ w
    if regression_type=="mse":
        y_pred = [1 if y > 0 else -1 for y in y_pred]
    elif regression_type=="lr":
        y_pred = [1 if y > 0.5 else -1 for y in y_pred]
    return y_pred


def predict_and_evaluate(x, w, y, regression_type="mse", verbose=1):
    """
    Make predictions and evaluate the model's performance.

    Parameters:
    x (array-like): Feature data for prediction.
    w (array-like): Weight parameters for the linear model.
    y (array-like): Ground truth labels for evaluation.
    regression_type (str, optional): Type of regression ('mse' for mean squared error, 'lr' for logistic regression). Default is 'mse'.
    verbose (int, optional): Verbosity level for evaluation. Default is 1.

    Returns:
    f1: F1-score
    """
    tx_te = np.c_[np.ones((x.shape[0], 1)), x]
    y_pred = tx_te @ w
    if regression_type=="mse":
        y_pred = [1 if y > 0 else -1 for y in y_pred]
    elif regression_type=="lr":
        y_pred = [1 if y > 0.5 else -1 for y in y_pred]
    f1 = evaluate(y_pred, y, verbose)
    return f1


def remove_uniquevalue_cols(x_train,x_test):
    std_train = np.nanstd(x_train, axis=0)
    non_zero_std_columns = np.where(std_train != 0)[0]
    return x_train[:, non_zero_std_columns], x_test[:, non_zero_std_columns]

def standardize(x_train, x_test):
    """
    Standardize the feature data for training and testing sets.

    Parameters:
    x_train (array-like): Training set feature data.
    x_test (array-like): Testing set feature data.

    Returns:
    tuple: A tuple containing the standardized training and testing sets.

    This function standardizes the feature data to have zero mean and unit variance
    for columns with non-zero standard deviation.
    """

    x_train_st = (x_train - np.nanmean(x_train, axis=0)) / np.nanstd(x_train, axis=0)
    x_test_st = (x_test - np.nanmean(x_train, axis=0)) / np.nanstd(x_train, axis=0)

    return x_train_st, x_test_st


def duplicate_1rows(x_train, y_train, n=3):
    """
    Duplicate rows with label 1 in the training data.

    Parameters:
    x_train (array-like): Training set feature data.
    y_train (array-like): Training set labels.
    n (int, optional): Number of times to duplicate rows with label 1. Default is 3.

    Returns:
    tuple: A tuple containing the duplicated training feature data and labels.

    This function duplicates rows in the training data where the label (y_train) is equal to 1.
    It allows to increase the representation of the minority label in this imbalanced dataset.
    """
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


def nan_to_mean(x_train, x_test):
    """
    Replace NaN values with column means in training and testing data.

    Parameters:
    x_train (array-like): Training set feature data.
    x_test (array-like): Testing set feature data.

    Returns:
    tuple: A tuple containing the modified training and testing feature data.

    This function replaces NaN values in the input data with the mean
    of the corresponding column. This is a data preprocessing step to handle
    missing values.
    """
    # Calculate the mean of each column while ignoring NaN values
    column_means = np.nanmean(x_train, axis=0)
    
    # Replace NaN values in x_train and x_test with the respective column means
    x_train_nonan = np.where(np.isnan(x_train), column_means, x_train)
    x_test_nonan = np.where(np.isnan(x_test), column_means, x_test)
    
    return x_train_nonan, x_test_nonan

def replace_dk_values_with_nan(x_train):
    """
    Replace specific response values with NaN in the input data.

    Parameters:
    x (array-like): Input data containing response values.

    Returns:
    array-like: Input data with specific values replaced by NaN.

    This function is designed to handle cases where certain response values indicate
    missing or ambiguous data, such as 'don't know' or 'refused to reply.' It identifies
    and replaces these values with NaN, making it easier to process the data.
    """
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

def evaluate(y_pred, y_gt, verbose=1):
    """
    Evaluate the performance of a classification model by comparing predicted labels to ground truth labels.

    Parameters:
    y_pred (array-like): Predicted labels from the classification model.
    y_gt (array-like): Ground truth labels for the corresponding data points.

    Returns:
    f1: The F1-score, a measure of the model's precision and recall.

    This function calculates and prints two performance metrics: accuracy and F1-score. 
    Accuracy is the ratio of correctly predicted labels to the total number of data points. 
    F1-score is a measure that combines both precision and recall, providing a single 
    metric for the model's overall performance.
    """
    result = y_pred == y_gt
    f1 = compute_f1(y_gt, y_pred)
    if verbose == 1:
        print(f'Accuracy: {list(result).count(True) / len(result)}')
        print(f'F1-score: {compute_f1(y_gt, y_pred)}')
    return f1


def build_poly(x, degree):
    """
    Build a polynomial feature matrix from the input data.

    Parameters:
    x (array-like): Input feature data.
    degree (int): Degree of the polynomial.

    Returns:
    array-like: Polynomial feature matrix.

    This function constructs a polynomial feature matrix from the input data, raising
    each feature to the power of 1 to 'degree'. It can be used to create higher-order
    polynomial features.
    """
    N, D =  x.shape
    poly = np.zeros((N, D*(degree)))

    for j in range(degree):
        poly[:, (D*j):((D*j)+D)] = x ** (j+1)

    return poly


def split_data(x, y, ratio, seed=1):
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