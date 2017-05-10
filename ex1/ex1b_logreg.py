from mlxtend.data import loadlocal_mnist
from sklearn.utils import shuffle
import os
import numpy as np
import time
import scipy.optimize as opt
from scipy.special import expit

def feature_normalize(train, test):
    """Normalize the features to have 0 mean and unit standard deviation

    @param train The features of training set
    @param test The features of the training set

    @return train, test The normalized features for both the training and testing
    """
    m = train.mean(axis=0)
    s = train.std(axis=0)
    train = train - m
    train = train / (s+0.1)
    test = test - m
    test = test / (s+0.1)
    return train, test

def load_data(img_dir, lab_dir):
    """Loads the MNIST data onto two numpy arrays and shuffles them

    @param img_dir The path where images are stored
    @param lab_dir The path where labels are stored

    @return X, y The Features and the labels
    """
    X,y = loadlocal_mnist(images_path=img_dir, labels_path=lab_dir)

    # Choose only y=0 and 1. Preprocess the data
    iy = np.in1d(y, [0, 1])
    ind = np.where(iy)[0].tolist()
    X = X[ind, :]
    y = y[ind]
    X,y = shuffle(X, y, random_state=0)
    return X, y

def sigmoid(z):
    """Calculate the sigmoid value of the function

    @param z The value to Calculate the sigmoid for

    @return the sigmoid of z
    """
    return expit(z)

def logCost(theta, X, y):
    """Cost function for the logistic regression, which has to be minimized

    @param theta The weights
    @param X The training set features
    @param y The training set labels(values)

    @return J The cost of the function
    """
    m, n = X.shape
    J = 0
    for i in range(m):
        t_xi = 0
        for j in range(n):
            t_xi += theta[j]*X[i, j]
        h_xi = sigmoid(t_xi)
        J = J - (y[i]*np.log(h_xi) + (1-y[i])*np.log(1-h_xi));
    return J

def logGrad(theta, X, y):
    """Gradient of the cost function for logistic regression

    @param theta The weights
    @param X The training set features
    @param y The training set labels(values)

    @return grad The gradient values
    """
    m, n = X.shape
    grad = np.zeros(theta.shape)
    h_x = np.zeros(y.shape)
    for i in range(m):
        t_xi = 0
        for j in range(n):
            t_xi += theta[j]*X[i, j]
        h_x[i] = sigmoid(t_xi)

    for j in range(n):
        for i in range(m):
            grad[j] += (h_x[i] - y[i])*X[i, j]
    return grad

def logCostVec(theta, X, y):
    """Vectorized cost function for logistic regression

    @param theta The weights
    @param X The training set features
    @param y The training set labels(values)

    @return J The cost of the function
    """
    h_x = sigmoid(X.dot(theta))
    J = sum(y*np.log(h_x) + (1-y)*np.log(1-h_x))
    J = -1*J
    return J

def logGradVec(theta, X, y):
    """Vectorized implementation of the gradient for the cost function

    @param theta The weights
    @param X The training set features
    @param y The training set labels(values)

    @return grad The gradient values
    """
    h_x = sigmoid(X.dot(theta))
    grad = np.transpose(X).dot(h_x - y)
    return grad

def bin_class_accuracy(theta, X, y):
    """Function to Calculate the accuracy of binary classification

    @param theta The weights
    @param X The features
    @param y The labels(values)

    @return acc The accuracy of classification
    """
    correct = sum(np.equal(y, (sigmoid(X.dot(theta)))>0.5))
    acc = correct*1.0/len(y)
    return acc

def main():
    # Load the data from the data file
    cur_dir = os.path.dirname(__file__)
    img_dir = os.path.join(cur_dir, '../data/train-images-idx3-ubyte')
    lab_dir = os.path.join(cur_dir, '../data/train-labels-idx1-ubyte')
    timg_dir = os.path.join(cur_dir, '../data/t10k-images-idx3-ubyte')
    tlab_dir = os.path.join(cur_dir, '../data/t10k-labels-idx1-ubyte')
    X, y = load_data(img_dir=img_dir, lab_dir=lab_dir)
    tX, ty = load_data(img_dir=timg_dir, lab_dir=tlab_dir)
    train_X, test_X = feature_normalize(train=X, test=tX)
    train_X = np.insert(train_X, 0, 1, axis=1)
    test_X = np.insert(test_X, 0, 1, axis=1)
    train_Y = y
    test_Y = ty
    print "Training set dimensions: %s x %s" % (train_X.shape)
    print "Training labels dimensions: %s" % (train_Y.shape)
    print "Testing set dimensions: %s x %s" % (test_X.shape)
    print "Testing labels dimensions: %s" % (test_Y.shape)
    m,n = train_X.shape

    """
    ### Non-vectorized native implementation, using for loops took around ###
    ### 39-40 minutes to converge ###
    ### It is hence NOT advisable to uncomment this section, which implements ###
    ### the optimization with non-vectoized cost and gradient functions ###

    # Initialize random weights
    theta = np.random.rand(n)*0.001

    # Perform the optimizations
    t0 = time.time()
    optTheta = opt.minimize(
    fun=logCost,
    x0=theta,
    args=(train_X, train_Y),
    method='L-BFGS-B', jac=logGrad,
    options={'maxiter' : 100, 'disp' : True})
    t1 = time.time()
    print "Optimization took %.7f seconds.\n" % (t1-t0)
    optTheta = optTheta.x

    # Print the training and testing accuracy
    tr_acc = bin_class_accuracy(optTheta, train_X, train_Y)
    print "Training accuracy: %2.2f%%\n" % (100.0*tr_acc)
    te_acc = bin_class_accuracy(optTheta, test_X, test_Y)
    print "Testing accuracy: %2.2f%%\n" % (100.0*te_acc)

    # Running the Vectorized code now
    print "-------------------------------------------------------------"
    print "\n\nVectorized codes start now\n\n"
    """
    # Initialize random weights
    theta = np.random.rand(n)*0.001

    # Perform the optimizations
    t0 = time.time()
    optTheta = opt.minimize(
    fun=logCostVec,
    x0=theta,
    args=(train_X, train_Y),
    method='L-BFGS-B', jac=logGradVec,
    options={'maxiter' : 100, 'disp' : True})
    t1 = time.time()
    print "Optimization took %.7f seconds.\n" % (t1-t0)
    optTheta = optTheta.x

    # Print the training and testing accuracy
    tr_acc = bin_class_accuracy(optTheta, train_X, train_Y)
    print "Training accuracy: %2.2f%%\n" % (100.0*tr_acc)
    te_acc = bin_class_accuracy(optTheta, test_X, test_Y)
    print "Testing accuracy: %2.2f%%\n" % (100.0*te_acc)

if __name__ == '__main__':
    main()
