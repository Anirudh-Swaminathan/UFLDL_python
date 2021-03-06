from mlxtend.data import loadlocal_mnist
from sklearn.utils import shuffle
import os
import numpy as np
import time
import scipy.optimize as opt
from random import randint
from mlxtend.preprocessing import one_hot

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
    X,y = shuffle(X, y, random_state=0)
    return X, y

def softCostVec(theta, X, y):
    """Vectorized cost function for softmax regression

    @param theta The weights
    @param X The training set features
    @param y The training set labels(values)

    @return J The cost of the function
    """
    m,n = X.shape
    num_classes = theta.shape[0]/n
    theta = np.reshape(a=theta, newshape=(n, num_classes))

    # Perform one-hot encoding for each of the m training examples
    ty = one_hot(y=y, num_labels=10)

    # Calculate the hypothesis
    epow = np.exp(X.dot(theta))
    #epow = np.insert(epow, epow.shape[1], 1, axis=1)
    h_x = epow / epow.sum(axis=1)[:, None]
    J = np.sum(np.multiply(ty, np.log(h_x)))
    J = -1.0 * J
    return J

def softGradVec(theta, X, y):
    """Vectorized implementation of the gradient for the cost function

    @param theta The weights
    @param X The training set features
    @param y The training set labels(values)

    @return grad The gradient values
    """
    m,n = X.shape
    num_classes = theta.shape[0]/n
    theta = np.reshape(a=theta, newshape=(n, num_classes))
    grad = np.zeros(theta.shape)

    # Perform one-hot encoding for each of the m training examples
    ty = one_hot(y=y, num_labels=10)

    # Calculate the hypothesis
    epow = np.exp(X.dot(theta))
    #epow = np.insert(epow, epow.shape[1], 1, axis=1)
    h_x = epow / epow.sum(axis=1)[:, None]
    # grad = -1.0 * np.transpose(X).dot(ty - epow)
    grad = -1.0 * np.transpose(X).dot(ty - h_x)
    #grad = np.delete(grad, grad.shape[1]-1, axis=1)
    grad = grad.flatten()
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
        Jtp = softCostVec(theta=tp, X=X, y=y)
        Jtm = softCostVec(theta=tm, X=X, y=y)
        g_est = (Jtp - Jtm) / (2.0 * epsilon)
        g_act = softGradVec(theta=t0, X=X, y=y)
        s_er += abs(g_act[j] - g_est)
    av_er = s_er*1.0/num_iters
    return av_er

def multi_class_accuracy(theta, X, y):
    """Function to Calculate the accuracy of Multiclass classification

    @param theta The weights
    @param X The features
    @param y The labels(values)

    @return acc The accuracy of classification
    """
    correct = np.sum(np.argmax(X.dot(theta), axis=1) == y)
    return correct*1.0 / y.size

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
    num_classes = 10

    # Initialize random weights
    theta = np.random.rand(n, num_classes)*0.001
    print theta.shape

    # Check the gradient before the optimization with the vectorized cost function
    erro = grad_check(t0=theta.flatten(), num_iters=100, X=train_X, y=train_Y)
    print "\nThe average error in the gradients is %.6f\n" % (erro)
    assert erro <= 1e-03, "Error in gradients is too much"

    # Perform the optimizations
    t0 = time.time()
    optTheta = opt.minimize(
    fun=softCostVec,
    x0=theta.flatten(),
    args=(train_X, train_Y),
    method='L-BFGS-B', jac=softGradVec,
    options={'maxiter' : 100, 'disp' : True})
    t1 = time.time()
    print "Optimization took %.7f seconds.\n" % (t1-t0)
    optTheta = optTheta.x

    # Reshape the theta to nxk
    optTheta = np.reshape(a=optTheta, newshape=(n, num_classes))

    # Use 0 for the final class to make theta nxk
    #optTheta = np.insert(optTheta, optTheta.shape[1], 0, axis=1)

    # Print the training and testing accuracy
    tr_acc = multi_class_accuracy(optTheta, train_X, train_Y)
    print "Training accuracy: %2.2f%%\n" % (100.0*tr_acc)
    te_acc = multi_class_accuracy(optTheta, test_X, test_Y)
    print "Testing accuracy: %2.2f%%\n" % (100.0*te_acc)

if __name__ == '__main__':
    main()
