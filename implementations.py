"""The 6 ML methods implemented in the labs"""
import numpy as np
from utils import *

# Set a random seed for reproducibility
np.random.seed(42)

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
    ws = [initial_w]
    losses = []
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_mse(y, tx, w)
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma*gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
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
            loss = compute_mse(minibatch_y, minibatch_tx, w)
            stoch_gradient += grad
            stoch_loss += loss
        
        # Update w by gradient
        w = w - gamma*stoch_gradient

        ws.append(w)
        losses.append(stoch_loss)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
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
    
    # Check if the determinant is close to zero
    det_XTX_reg = np.linalg.det(XTX_reg)
    if det_XTX_reg < 1e-10:
        # Add a small value to the diagonal to make it non-singular
        XTX_reg += np.identity(tx.shape[1]) * 1e-6
    
    w = np.linalg.solve(XTX_reg, XTY)

    mse = compute_mse(y, tx, w)

    return w, mse


# Method 5: Logistic regression using gradient descent
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
    """
    w = initial_w
    loss = 0
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = gradient_descent_for_logistic_regression(y, tx, w, gamma)
        print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    print("loss={l}".format(l=loss))

    return w, loss


# Method 6: Regularized logistic regression using gradient descent
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        lambda_: scalar.
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
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
    print("Final loss={l}".format(l=loss))

    return w, loss
