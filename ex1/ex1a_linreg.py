import numpy as np
import time
import scipy.optimize as opt
import matplotlib.pyplot as plt

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

def main():
    # Load the data from housing.data
    data = np.genfromtxt(fname="housing.data") #, delimiter=(7, 5, 6, 1, 6, 6, 6, 7, 2, 5, 5, 6, 5, 5))
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
    plt.savefig('ex1a_linreg.png', bbox_inches='tight')
    plt.show()
    print "Finished plotting the graph"

if __name__ == '__main__':
    main()
