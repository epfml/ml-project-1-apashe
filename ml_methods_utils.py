"""Functions used in the ML methods"""
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

def compute_mse_linear_regression(y, tx, w):
    """compute the loss by mse for linear regression
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

def compute_gradient_linear_regression(y, tx, w):
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


def calculate_loss_logistic_regression(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    y = y.reshape(-1,1)
    N = y.shape[0]

    return -np.mean(y*np.log(sigmoid(tx@w))+(np.ones((N,1))-y)*np.log(np.ones((N,1))-sigmoid(tx@w)))



def calculate_gradient_logistic_regression(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """

    N = len(y)
    y = y.reshape(-1,1)
    # ***************************************************
    return (1/N)*(tx.T)@((sigmoid(tx@w))-(y))


def learning_by_gradient_descent_logistic(y, tx, w, gamma):
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
    gradient = calculate_gradient_logistic_regression(y, tx, w)
    loss = calculate_loss_logistic_regression(y, tx, w)
    w_next = w - gamma * gradient
    return loss, w_next


def compute_loss_and_grad_penalized_logistic(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
    """
    loss_logistic = calculate_loss_logistic_regression(y, tx, w)
    loss_regularization = 0.5 * lambda_ * np.sum(w**2)
    loss = loss_logistic + 2 * loss_regularization
    
    gradient_logistic = calculate_gradient_logistic_regression(y, tx, w)
    gradient_regularization = lambda_ * w
    gradient = gradient_logistic + 2 * gradient_regularization
    
    return loss, gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
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
    loss, gradient = compute_loss_and_grad_penalized_logistic(y, tx, w, lambda_)
    w = w - gamma * gradient
    return loss, w



def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    # ***************************************************
    # INSERT YOUR CODE HERE
    e = y - tx@w
    N = y.size
    gradient = -(1/N)*(tx.T @ e)
    # ***************************************************
    return gradient


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
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
        # ***************************************************
        stoch_gradient = [0,0]
        stoch_loss = 0
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = compute_stoch_gradient(minibatch_y,minibatch_tx,w)
            loss = calculate_loss_logistic_regression(minibatch_y, minibatch_tx, w)
            loss = compute_mse_linear_regression(minibatch_y, minibatch_tx, w)
            # loss = compute_loss(minibatch_y, minibatch_tx, w)
            stoch_gradient += grad
            stoch_loss += loss
        # ***************************************************
        w = w - gamma*stoch_gradient

        ws.append(w)
        losses.append(stoch_loss)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, ws

