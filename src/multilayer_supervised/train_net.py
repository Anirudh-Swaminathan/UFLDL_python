from mlxtend.data import loadlocal_mnist
import os
import numpy as np
from mlxtend.preprocessing import one_hot
import time
import scipy.optimize as opt
from random import randint
from scipy.stats import hypsecant as sechfun

def initialize_weights(ei):
    """Function to randomly initialize the weights of all the layers

    @param ei The hyper-parameters for the neural network to be trained

    @return ret The list of dicts, each dict containing the matrix W
                and the vector b for that layer
    """
    ret = []
    for l in range(len(ei['layer_sizes'])):
        if l>0:
            prev_size = ei['layer_sizes'][l-1]
        else:
            prev_size = ei['input_dim']
        cur_size = ei['layer_sizes'][l]
        # Xavier's Scaling factor
        s = 6**0.5 / (cur_size + prev_size)**0.5
        d = {}
        ret.append(d)
        ret[l]['W'] = np.random.uniform(low=0.0, high=1.0, size=(cur_size, prev_size))*2*s - s
        ret[l]['b'] = np.zeros(shape=(cur_size, 1))
    return ret

def unroll_stack(stac):
    """Function to unroll the stack into a list of parameters

    @param stac The stack to be unrolled into a 1D list of parameters

    @return retun The unrolled list of parameters
    """
    retun = []
    for d in range(len(stac)):
        retun.extend(stac[d]['W'].flatten())
        retun.extend(stac[d]['b'].flatten())
    #print len(retun)
    return np.array(retun)

def restack(params, ei):
    """Function to stack up the unrolled parameters

    @param params The unrolled parameters to be stacked
    @param ei The network configuration

    @return retsta The stacked params
    """
    num_layers = len(ei['layer_sizes'])
    retsta = []
    prev_size = ei['input_dim']
    cur_pos = 0
    for d in range(num_layers):
        di = {}
        retsta.append(di)
        num_units = ei['layer_sizes'][d]

        # Extract the weights
        wlen = num_units*prev_size
        retsta[d]['W'] = np.reshape(params[cur_pos:cur_pos+wlen], (num_units, prev_size))
        cur_pos += wlen

        # Extract the bias terms
        blen = num_units
        retsta[d]['b'] = np.reshape(params[cur_pos:cur_pos+blen], (num_units, 1))
        cur_pos += blen
        prev_size = num_units
    #print num_layers, len(retsta), retsta[0]['W'].shape, retsta[0]['b'].shape, retsta[1]['W'].shape, retsta[1]['b'].shape
    return retsta

def actFun(z, ac):
    """Function to Calculate the activations of different units

    @param z the parameter of the function
    @param ac The function to calculate for

    @return The final activation
    """
    if ac == 'logistic':
        #print sigmoid(z)[:2][:2]
        return sigmoid(z)
    elif ac == 'tanh':
        #print np.tanh(z)[:2][:2]
        return np.tanh(z)
    else:
        #print np.maximum(np.zeros(z.shape), z)[:2][:2]
        return np.maximum(np.zeros(z.shape), z)

def safe_log(x, nan_substitute=-1e+4):
    l = np.log(x)
    l[np.logical_or(np.isnan(l), np.isinf(l))] = nan_substitute
    return l

def der(z, ac):
    """Function to calculate the derivative of the desired activation Function

    @param z The parameter of the function
    @param ac The activation function type

    @return The final derivative
    """
    if ac=='logistic':
        #print sigmoid(z)*(1-sigmoid(z))[:2][:2]
        return sigmoid(z)*(1-sigmoid(z))
    elif ac=='tanh':
        #print sechfun(x)**2.0[:2][:2]
        return sechfun(z)**2.0
    else:
        #print np.maximum(np.zeros(z.shape), z>0)[:2][:2]
        return np.maximum(np.zeros(z.shape), z>0)

def forwardProp(unroll_theta, ei, X, y):
    """Function to perform forward propagation

    @param unroll_theta The unrolled theta
    @param ei The network configuration
    @param X The training samples
    @param y the training labels

    @return hAct the activation of all layers
    """
    stack = restack(unroll_theta, ei=ei)
    #print len(stack)
    numHidden = len(ei['layer_sizes']) - 1
    #print numHidden
    hAct = []
    for l in range(numHidden+1):
        if l==0:
            xh = np.dot(stack[l]['W'], np.transpose(X))
            xh = xh + np.reshape(stack[l]['b'], (stack[l]['b'].shape[0], 1))
            hAct.append(actFun(xh, ei['activation_fun']))
        elif l==numHidden:
            xh = np.dot(stack[l]['W'], hAct[l-1])
            xh = xh + np.reshape(stack[l]['b'], (stack[l]['b'].shape[0], 1))
            hwb = np.exp(xh)
            hAct.append(hwb / map(float, sum(hwb)))
        else:
            xh = np.dot(stack[l]['W'], hAct[l-1])
            xh = xh + np.reshape(stack[l]['b'], (stack[l]['b'].shape[0], 1))
            hAct.append(actFun(xh, ei['activation_fun']))
    #print len(hAct)
    return hAct

def costFunc(unroll_theta, ei, X, y):
    """Function to calculate the cost of the neural network

    @param unroll_theta The unrolled theta
    @param ei The network configuration
    @param X The training samples
    @param y The training labels

    @return J The cost of the neural network
    """
    stack = restack(unroll_theta, ei=ei)
    #print len(stack)
    numHidden = len(ei['layer_sizes']) - 1
    #print numHidden
    hAct = forwardProp(unroll_theta=unroll_theta, ei=ei, X=X, y=y)
    #print hAct[0].shape
    #print hAct[1].shape
    #print hAct[:5][:5]
    ty = one_hot(y=y, num_labels=10)
    ty = np.transpose(ty)
    m = X.shape[0]
    #print m
    #print ty.shape
    #print hAct[-1].shape
    crossEntropyCost = -1.0 * sum(sum(ty * safe_log(hAct[-1]))) / m
    regCost = 0
    for i in range(len(ei['layer_sizes'])):
        regCost += ei['lambda'] / 2 * sum(sum(stack[i]['W']**2.0))
    #print crossEntropyCost, regCost
    J = crossEntropyCost + regCost
    return J

def gradFunc(unroll_theta, ei, X, y):
    """Function to calculate the gradient of the cost Function

    @param unroll_theta The unrolled parameters of the neural network
    @param ei The network configuration
    @param X The training samples
    @param y The training labels

    @return grad The unrolled gradient
    """
    stack = restack(unroll_theta, ei=ei)
    numHidden = len(ei['layer_sizes']) - 1
    hAct = forwardProp(unroll_theta=unroll_theta, ei=ei, X=X, y=y)
    ty = one_hot(y=y, num_labels=10)
    ty = np.transpose(ty)
    sDel = [None] * (numHidden+1)
    m = X.shape[0]
    for i in range(numHidden, -1, -1):
        if i==numHidden:
            fi = -1.0 * (ty - hAct[i])
            sDel[i] = fi
        elif i==0:
            znl = np.dot(stack[i]['W'], np.transpose(X))
            znl = znl + np.reshape(stack[i]['b'], (stack[i]['b'].shape[0], 1))
            de = der(znl, ei['activation_fun'])
            sDel[i] = np.dot(np.transpose(stack[i+1]['W']), sDel[i+1]) * de
        else:
            znl = np.dot(stack[i]['W'], hAct[i-1])
            znl = znl + np.reshape(stack[i]['b'], (stack[i]['b'].shape[0], 1))
            de = der(znl, ei['activation_fun'])
            sDel[i] = np.dot(np.transpose(stack[i+1]['W']), sDel[i+1]) * de
    gradStack = []
    for i in range(numHidden+1):
        ap = {}
        gradStack.append(ap)
        if i==0:
            gradStack[i]['W'] = 1.0 / m * np.dot(sDel[i], X) + ei['lambda']*stack[i]['W']
            gradStack[i]['b'] = 1.0 / m * np.sum(sDel[i], 1)
        else:
            gradStack[i]['W'] = 1.0 / m * np.dot(sDel[i], np.transpose(hAct[i-1])) + ei['lambda']*stack[i]['W']
            gradStack[i]['b'] = 1.0 / m * np.sum(sDel[i], 1)
    grad = unroll_stack(gradStack)
    return grad

def neural_net_acc(optTheta, ei, X, y):
    """Function to calculate the accuracy of the training and testing sets

    @param optTheta The trained unrolled weights
    @param ei The network configuration
    @param X The training/testing data
    @param y The training/testing labels

    @return acc The accuracy of the neural network
    """
    hAc = forwardProp(unroll_theta=optTheta, ei=ei, X=X, y=y)
    pred = hAc[-1]
    pred = np.argmax(a=pred, axis=0)
    correct = np.sum(pred == y)
    return correct*1.0 / y.size

def grad_check(t0, ei, X, y, num_iters=30):
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
        Jtp = costFunc(tp, ei, X=X, y=y)
        Jtm = costFunc(tm, ei, X=X, y=y)
        g_est = (Jtp - Jtm) / (2.0 * epsilon)
        g_act = gradFunc(t0, ei, X=X, y=y)
        s_er += abs(g_act[j] - g_est)
    av_er = s_er*1.0/num_iters
    return av_er 

def main():
    # Load the data from the data file
    cur_dir = os.path.dirname(__file__)
    img_dir = os.path.join(cur_dir, '../data/train-images-idx3-ubyte')
    lab_dir = os.path.join(cur_dir, '../data/train-labels-idx1-ubyte')
    timg_dir = os.path.join(cur_dir, '../data/t10k-images-idx3-ubyte')
    tlab_dir = os.path.join(cur_dir, '../data/t10k-labels-idx1-ubyte')
    X, y = loadlocal_mnist(images_path=img_dir, labels_path=lab_dir)
    tX, ty = loadlocal_mnist(images_path=timg_dir, labels_path=tlab_dir)

    #### Inputting the parameters of the neural network ####
    ei = {}
    ei['input_dim'] = 784
    ei['output_dim'] = 10
    ei['layer_sizes'] = []
    try:
        a = raw_input("Enter space separated hidden layer sizes\n")
        if a == "":
            a="256"
        l = a.split()
        il = map(int, l)
        #print il
        ei['layer_sizes'].extend(il)
        #print ei['layer_sizes']
    except Exception as e:
        print "\nInvalid input for layer sizes. Initializing default layer sizes\n"
        ei['layer_sizes'].append(256)
    ei['layer_sizes'].append(ei['output_dim'])
    try:
        ei['lambda'] = float(raw_input("Enter the regularization parameter(between 0 and 1): "))
    except Exception as e:
        print "\nInvalid regularization parameter. No regularization will be implemented\n"
        ei['lambda'] = 0
    if ei['lambda']<0 or ei['lambda']>1:
        ei['lambda'] = 0
    try:
        cho = int(raw_input("Enter integer choice of activation for neuron\n1. Sigmoid\n2. tanh\n3. reLU\n"))
    except Exception as e:
        print "\nError in input. Going with default reLU\n"
        cho = 3
    if cho not in [1, 2, 3]:
        print "\nError invalid input parameter. Going with default reLU\n"
        cho = 3
    if cho == 1:
        ei['activation_fun'] = 'logistic'
    elif cho == 2:
        ei['activation_fun'] = 'tanh'
    else:
        ei['activation_fun'] = 'reLU'

    print "The configuration of the network is as follows\n", ei

    # Randomly initialize the weights for all the layers
    stack = initialize_weights(ei=ei)
    params = unroll_stack(stac=stack)

    print "The initial cost is ",costFunc(unroll_theta=params, ei=ei, X=X, y=y)
    print "The first and the last 5 grads are"
    gt = gradFunc(unroll_theta=params, ei=ei, X=X, y=y)
    print gt[:5]," ",gt[-5:]

    # Check the gradient before the optimization with the vectorized cost function
    erro = grad_check(t0=params, ei=ei, num_iters=8, X=X, y=y)
    print "\nThe average error in the gradients is %.8f\n" % (erro)
    assert erro <= 1e-03, "Error in gradients is too much"
    _ = raw_input("Press ENTER to continue")

    # Perform the optimizations
    t0 = time.time()
    opt_params = opt.minimize(
    fun=costFunc,
    x0=params,
    args=(ei, X, y),
    method='L-BFGS-B',
    #method='SLSQP', Doesnt work
    #method='COBYLA', Doesnt work
    #method='TNC',Doesnt work
    #method='BFGS',Doesnt work
    #method='Newton-CG', # Torturously slow
    jac=gradFunc,
    options={'disp' : True, 'maxiter' : 200})
    """
    opt_params = opt.fmin_l_bfgs_b(
    func=costFunc,
    x0=params,
    fprime=gradFunc,
    args=(ei, X, y),
    disp=True,
    maxfun=1000000)
    """
    t1 = time.time()
    print "Optimization took %.7f seconds.\n" % (t1-t0)
    opt_params = opt_params.x

    # Print the training and testing accuracy
    tr_acc = neural_net_acc(optTheta=opt_params, ei=ei, X=X, y=y)
    print "Training accuracy: %2.2f%%\n" % (100.0*tr_acc)
    te_acc = neural_net_acc(optTheta=opt_params, ei=ei, X=tX, y=ty)
    print "Testing accuracy: %2.2f%%\n" % (100.0*te_acc)


if __name__ == '__main__':
    main()
