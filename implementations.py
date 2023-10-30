"""The 6 ML methods implemented in the labs"""
import numpy as np
from utils import *

# Set a random seed for reproducibility
np.random.seed(1)

# Method 1: Linear regression using gradient descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
    """
    # Define parameters to store w and loss
    w = initial_w
    losses = []
    loss = compute_mse(y, tx, w)
    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma*gradient
        loss = compute_mse(y, tx, w)


        print(
            "GD iter. {bi}/{ti}: loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss
            )
        )
    return w, loss


# Method 2: Linear regression using stochastic gradient descent
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
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
        stoch_gradient = np.zeros(tx.shape[1])
        stoch_loss = 0
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1):
            grad = compute_gradient_mse(minibatch_y,minibatch_tx,w)
            stoch_gradient += grad
        
        # Update w by gradient
        w = w - gamma*stoch_gradient

        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1):
            loss = compute_mse(minibatch_y, minibatch_tx, w)
            stoch_loss += loss
    
        ws.append(w)
        losses.append(stoch_loss)

        print(
            "SGD iter. {bi}/{ti}: loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=stoch_loss
            )
        )
    return w, stoch_loss


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
    mse = compute_mse(y, tx, w)
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

    w = np.linalg.solve(XTX_reg, XTY)
    mse = compute_mse(y, tx, w)

    return w, mse


# Method 5: Logistic regression using gradient descent
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """return the loss and gradient.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
    """
    w = initial_w
    loss = compute_logistic_loss(y, tx, w)
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = gradient_descent_for_logistic_regression(y, tx, w, gamma)
        print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

    return w, loss


# Method 6: Regularized logistic regression using gradient descent
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """return the loss and gradient.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        lambda_: scalar.
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
    """
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get penalized loss and update w.
        loss_pen, w = gradient_descent_for_penalized_logistic_regression(y, tx, w, gamma, lambda_)
        print("Current iteration={i}, loss={l}".format(i=iter, l=loss_pen))
    loss = compute_logistic_loss(y, tx, w)

    return w, loss


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
            gradient_logistic = compute_gradient_logistic_loss(minibatch_y, minibatch_tx, w)
            gradient_regularization = lambda_ * w
            gradient = gradient_logistic + 2 * gradient_regularization

            stoch_gradient += gradient
        w = w - gamma*stoch_gradient

        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            loss_logistic = compute_logistic_loss(minibatch_y, minibatch_tx, w)
            loss_regularization = 0.5 * lambda_ * np.sum(w**2)
            loss = loss_logistic + 2 * loss_regularization  
            stoch_loss += loss
        print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    print("loss_best={l}".format(l=loss_best))

    return w, stoch_loss