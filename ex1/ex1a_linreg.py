import numpy as np
import time
import scipy.optimize as opt
import matplotlib.pyplot as plt
import os
from random import randint

def linCost(theta, X, y):
    """Cost function for the linear regression, which has to be minimized

    @param theta The weights
    @param X The training set features
    @param y The training set labels(values)

    @return J The cost of the function
    """
    m, n = X.shape
    J = 0
    for i in range(m):
        h_xi = 0
        for j in range(n):
            h_xi += theta[j]*X[i, j]
        J += 0.5 * ((h_xi - y[i])**2)
    return J

def linGrad(theta, X, y):
    """Gradient of the cost function for linear regression

    @param theta The weights
    @param X The training set features
    @param y The training set labels(values)

    @return grad The gradient values
    """
    m, n = X.shape
    grad = np.zeros(theta.shape)
    h_x = np.zeros(y.shape)
    for i in range(m):
        h_xi = 0
        for j in range(n):
            h_xi += theta[j]*X[i, j]
        h_x[i] = h_xi

    for j in range(n):
        for i in range(m):
            grad[j] += (h_x[i] - y[i])*X[i, j]
    return grad

def linCostVec(theta, X, y):
    """Vectorized cost function for linear regression

    @param theta The weights
    @param X The training set features
    @param y The training set labels(values)

    @return J The cost of the function
    """
    h_x = X.dot(theta)
    J = 0.5 * sum((h_x - y.flatten()) ** 2)
    return J

def linGradVec(theta, X, y):
    """Vectorized implementation of the gradient for the cost function

    @param theta The weights
    @param X The training set features
    @param y The training set labels(values)

    @return grad The gradient values
    """
    h_x = X.dot(theta)
    grad = np.transpose(X).dot(h_x - y.flatten())
    return grad

def grad_check(t0, X, y, num_iters=30):
    """Check the gradient with numerically computed one, and return average error

    @param t0 The initial theta value
    @param X The training set features
    @param y The training set labels(values)

    @return av_er The average error in the computation of the gradient
    """
    epsilon = 10**-4
    s_er = 0
    n = t0.shape[0]
    for i in range(num_iters):
        j = randint(0, n-1)
        tp = np.copy(t0)
        tm = np.copy(t0)
        tp[j] = tp[j] + epsilon
        tm[j] = tm[j] - epsilon
        Jtp = linCostVec(theta=tp, X=X, y=y)
        Jtm = linCostVec(theta=tm, X=X, y=y)
        g_est = (Jtp - Jtm) / (2.0 * epsilon)
        g_act = linGradVec(theta=t0, X=X, y=y)
        s_er += abs(g_act[j] - g_est)
    av_er = s_er*1.0/num_iters
    return av_er

def main():
    dir_name = os.path.dirname(__file__)
    file_name = os.path.join(dir_name, '../data/housing.data')
    # Load the data from housing.data
    data = np.genfromtxt(fname=file_name) #, delimiter=(7, 5, 6, 1, 6, 6, 6, 7, 2, 5, 5, 6, 5, 5))
    #print data[:10, :]

    # Insert the x0 term (=1) for all the m training examples
    data = np.insert(data, 0, 1, axis=1)

    # Randomly shuffle the data
    np.random.shuffle(data)
    print data.shape

    # Perform the train and test split
    train_X = data[:400, :-1]
    train_Y = data[:400, -1:]
    test_X = data[400:, :-1]
    test_Y = data[400:, -1:]
    print train_X.shape, train_Y.shape, test_X.shape, test_Y.shape
    #print data[:10, :]
    m = train_X.shape[0]
    n = train_X.shape[1]

    # Initialize random weights
    theta = np.random.rand(n)

    # Perform the optimizations
    t0 = time.time()
    optTheta = opt.minimize(
    fun=linCost,
    x0=theta,
    args=(train_X, train_Y),
    method='bfgs', jac=linGrad,
    options={'maxiter' : 200, 'disp' : True})
    t1 = time.time()
    print "Optimization took %.7f seconds.\n" % (t1-t0)
    optTheta = optTheta.x

    # Print the training and testing error
    for dataset, (X, y) in (('train', (train_X, train_Y)), ('test', (test_X, test_Y))):
        act_prices = y.flatten()
        pred_prices = X.dot(optTheta).flatten()
        err = np.sqrt(np.mean((pred_prices - act_prices)**2))
        print "RMS",dataset,"error: %.6f" % err

    # Repeat the above procedure for the Vectorized implementation
    # Initialize random weights
    theta = np.random.rand(n)

    # Check the gradient before the optimization with the vectorized cost function
    erro = grad_check(t0=theta, num_iters=100, X=train_X, y=train_Y)
    print "\nThe average error in the gradients is %.6f\n" % (erro)

    # Perform the optimizations
    t0 = time.time()
    optTheta = opt.minimize(
    fun=linCostVec,
    x0=theta,
    args=(train_X, train_Y),
    method='bfgs', jac=linGradVec,
    options={'maxiter' : 200, 'disp' : True})
    t1 = time.time()
    print "Optimization took %.7f seconds.\n" % (t1-t0)
    optTheta = optTheta.x

    # Print the training and testing error
    for dataset, (X, y) in (('train', (train_X, train_Y)), ('test', (test_X, test_Y))):
        act_prices = y.flatten()
        pred_prices = X.dot(optTheta).flatten()
        err = np.sqrt(np.mean((pred_prices - act_prices)**2))
        print "RMS",dataset,"error: %.6f" % err

    # Plot and save the data points
    plt.figure(figsize=(10, 8))
    plt.scatter(np.arange(test_Y.size), sorted(test_Y), c='r', alpha=0.5, marker='x', label="actual")
    plt.scatter(x=np.arange(test_Y.size), y=sorted(test_X.dot(optTheta).flatten()), c='b', alpha=0.5, marker='x',label="predicted")
    plt.legend(loc='upper left')
    plt.ylabel('House price ($1000s)')
    plt.xlabel('House #')
    op_f_name = os.path.join(dir_name, '../outputs/ex1a_linreg.png')
    plt.savefig(op_f_name, bbox_inches='tight')
    plt.show()
    print "Finished plotting the graph"

if __name__ == '__main__':
    main()
