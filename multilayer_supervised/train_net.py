from mlxtend.data import loadlocal_mnist
import os
import numpy as np
from mlxtend.preprocessing import one_hot
import time
import scipy.optimize as opt

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
    return retsta

def actFun(z, ac):
    """Function to Calculate the activations of different units

    @param z the parameter of the function
    @param ac The function to calculate for

    @return The final activation
    """
    if ac == 'logistic':
        return sigmoid(z)
    elif ac == 'tanh':
        return np.tanh(z)
    else:
        return np.maximum(np.zeros(z.shape), z)

def forwardProp(unroll_theta, ei, X, y):
    """Function to perform forward propagation

    @param unroll_theta The unrolled theta
    @param ei The network configuration
    @param X The training samples
    @param y the training labels

    @return hAct the activation of all layers
    """
    stack = restack(unroll_theta, ei=ei)
    numHidden = len(ei['layer_sizes']) - 1
    hAct = []
    for i in range(numHidden+1):
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
    numHidden = len(ei['layer_sizes']) - 1
    hAct = forwardProp(unroll_theta=unroll_theta, ei=ei, X=X, y=y)
    ty = one_hot(y=y, num_labels=10)
    ty = np.transpose(ty)
    m = X.shape[0]
    crossEntropyCost = -1.0 * sum(sum(ty * np.log(hAct[-1]))) / m
    regCost = 0
    for i in range(len(ei['layer_sizes'])):
        regCost += ei['lambda'] / 2 * sum(sum(stack[i]['W']**2.0))
    J = crossEntropyCost + regCost
    return J

def gradFunc():
    pass

def main():
    # Load the data from the data file
    cur_dir = os.path.dirname(__file__)
    img_dir = os.path.join(cur_dir, '../data/train-images-idx3-ubyte')
    lab_dir = os.path.join(cur_dir, '../data/train-labels-idx1-ubyte')
    timg_dir = os.path.join(cur_dir, '../data/t10k-images-idx3-ubyte')
    tlab_dir = os.path.join(cur_dir, '../data/t10k-labels-idx1-ubyte')
    X, y = loadlocal_mnist(img_dir=img_dir, lab_dir=lab_dir)
    tX, ty = loadlocal_mnist(img_dir=timg_dir, lab_dir=tlab_dir)

    #### Inputting the parameters of the neural network ####
    ei = {}
    ei['input_dim'] = 784
    ei['output_dim'] = 10
    try:
        ei['layer_sizes'] = map(int, raw_input.split("Enter space separated hidden layer sizes"))
    except Exception as e:
        ei['layer_sizes'] = [256]
    ei['layer_sizes'].append(ei['output_dim'])
    try:
        ei['lambda'] = float(raw_input("Enter the regularization parameter(between 0 and 1)"))
    except Exception as e:
        ei['lambda'] = 0
    if ei['lambda']<0 or ei['lambda']>1:
        ei['lambda'] = 0
    try:
        cho = int(raw_input("Enter integer choice of activation for neuron\n1. Sigmoid\n2. tanh\n3. reLU"))
    except Exception as e:
        cho = 3
    if cho not in [1, 2, 3]:
        cho = 3
    if cho == 1:
        ei['activation_fun'] = 'logistic'
    elif cho == 2:
        ei['activation_fun'] = 'tanh'
    else:
        ei['activation_fun'] = 'reLU'

    # Randomly initialize the weights for all the layers
    stack = initialize_weights(ei=ei)
    params = unroll_stack(stac=stack)

if __name__ == '__main__':
    main()
