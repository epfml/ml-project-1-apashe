"""The 6 ML methods implemented in the labs"""
import numpy as np
from ml_methods_utils import *
from sklearn.metrics import f1_score
from utils import evaluate1

# Method 1: Linear regression using gradient descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma, verbose=1):
# Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_mse_linear_regression(y, tx, w)
        gradient = compute_gradient_linear_regression(y, tx, w)
        w = w - gamma*gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)
        if verbose:
            print(
                "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
                )
            )
    return w, loss


# Method 2: Linear regression using stochastic gradient descent
def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma, y_test_split, tx_te):
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
    w_best = w
    f1_best = 0
    f1 = 0
    loss_best = 10**10

    for n_iter in range(max_iters):
        stoch_gradient = np.zeros(tx.shape[1])
        stoch_loss = 0
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = compute_gradient_linear_regression(minibatch_y,minibatch_tx,w)
            loss = compute_mse_linear_regression(minibatch_y, minibatch_tx, w)
            stoch_gradient += grad
            stoch_loss += loss
        
        w = w - gamma*stoch_gradient

        y_pred1 = tx_te @ w
        y_pred = [1 if y > 0 else -1 for y in y_pred1]
        f1 = f1_score(y_test_split, y_pred)
        
        if f1_best < f1:
            f1_best = f1
            loss_best = stoch_loss
            w_best = w
            print(f1)
        else:
            print("no")

        w = w_best

        ws.append(w)
        losses.append(stoch_loss)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return w_best, loss_best, f1_best


# Method 3: Least squares regression using normal equations
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    XTX = tx.T @ tx
    XTY = tx.T @ y
    w = np.linalg.solve(XTX, XTY)
    mse = compute_mse_linear_regression(y, tx, w)
    return w, mse


# Method 4: Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    """Implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N, D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape (D,), D is the number of features.
    """
    N = len(y)
    XTX = tx.T @ tx
    XTY = tx.T @ y

    lambdaI = 2 * N * lambda_ * np.identity(tx.shape[1])
    XTX_reg = XTX + lambdaI
    
    # Check if the determinant is close to zero
    det_XTX_reg = np.linalg.det(XTX_reg)
    if det_XTX_reg < 1e-10:
        # Add a small value to the diagonal to make it non-singular
        XTX_reg += np.identity(tx.shape[1]) * 1e-6
    
    w = np.linalg.solve(XTX_reg, XTY)

    mse = compute_mse_linear_regression(y, tx, w)

    return w, mse


# Method 5a: Logistic regression using gradient descent
def logistic_regression_gd(y, tx, initial_w, max_iters, gamma, verbose=1):
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent_logistic(y, tx, w, gamma)
        if verbose:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    if verbose:
        print("loss={l}".format(l=loss))

    return w, loss

# Method 5b: Logistic regression using SGD 
def logistic_regression_sgd(y, tx, initial_w, max_iters, gamma, batch_size):
    w = initial_w
    w_best = w
    loss_best = 10**20
    # start the logistic regression
    for iter in range(max_iters):
        stoch_gradient = np.zeros((tx.shape[1],1))
        stoch_loss = 0
        previous_loss = stoch_loss
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # get loss and update w.    
            gradient = calculate_gradient_logistic_regression(minibatch_y, minibatch_tx, w)
            loss = calculate_loss_logistic_regression(minibatch_y, minibatch_tx, w)
            stoch_gradient += gradient
            stoch_loss += loss
        w = w - gamma*stoch_gradient

        if stoch_loss < loss_best:
            w_best = w
            loss_best = stoch_loss

        print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    print("loss_best={l}".format(l=loss_best))

    return w_best, loss_best



# Method 6a: Regularized logistic regression using gradient descent
def reg_logistic_regression_gd(y, tx, lambda_, initial_w, max_iters, gamma, verbose=1):
    w = initial_w    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        if verbose:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    if verbose:
        print("loss={l}".format(l=loss))

    return w, loss


# Method 6b: Regularized logistic regression using SGD
def reg_logistic_regression_sgd(y, tx, lambda_, initial_w, max_iters, gamma, batch_size):
    w = initial_w
    w_best = w
    loss_best = 10**20
    # start the logistic regression
    for iter in range(max_iters):
        stoch_gradient = np.zeros((tx.shape[1],1))
        stoch_loss = 0
        previous_loss = stoch_loss
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # get loss and update w.    
            loss, gradient = compute_loss_and_grad_penalized_logistic(minibatch_y, minibatch_tx, w, lambda_)
            stoch_gradient += gradient
            stoch_loss += loss
        w = w - gamma*stoch_gradient

        if stoch_loss < loss_best:
            w_best = w
            loss_best = stoch_loss

        print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    print("loss_best={l}".format(l=loss_best))

    return w_best, loss_best





# Method 6b: Regularized logistic regression using SGD
def reg_logistic_regression_sgd_ev(y, tx, lambda_, initial_w, max_iters, gamma, batch_size, tx_te, y_test_split):
    w = initial_w
    w_best = w
    f1_best = 0
    # start the logistic regression
    for iter in range(max_iters):
        stoch_gradient = np.zeros((tx.shape[1],1))
        stoch_loss = 0
        previous_loss = stoch_loss
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # get loss and update w.    
            loss, gradient = compute_loss_and_grad_penalized_logistic(minibatch_y, minibatch_tx, w, lambda_)
            stoch_gradient += gradient
            stoch_loss += loss
        w = w - gamma*stoch_gradient

        y_pred1 = tx_te@w
        y_pred = [1 if y > 0.5 else -1 for y in y_pred1]

        acc,f1 = evaluate1(y_pred,y_test_split)


        if f1 > f1_best:
            f1_best = f1
            acc_best = acc
            w_best = w
            loss_best = stoch_loss

        print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    print("f1_best={l}".format(l=f1_best))

    return w_best, loss_best, f1_best, acc_best


