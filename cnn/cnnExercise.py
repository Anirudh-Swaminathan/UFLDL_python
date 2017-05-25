from mlxtend.data import loadlocal_mnist
import os
import numpy as np
from scipy.special import expit
from scipy.signal import convolve2d as conv2
from random import randint as randi

def sigmoid(z):
    """Calculate the sigmoid value of the function

    @param z The value to Calculate the sigmoid for

    @return the sigmoid of z
    """
    return expit(z)

def cnnConvolve(filterDim, numFilters, images, W, b):
    """ Function to return the convolution of the features given by W and b with
        the given images

    @param filterDim Filter(feature) dimensions
    @param numFilters Number of feature maps
    @param images large images to convolve with; Matrix in the form
           images(r, c, numImages)
    @param W Feature map of shape (filterDim, filterDim, numFilters)
    @param b Feature map of shape (numFilters)

    @return convolvedFeatures Matrix of convolved features in the form
                convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
    """
    imageDim, _, numImages = images.shape
    convDim = imageDim - filterDim + 1
    convolvedFeatures = np.zeros(shape=(convDim, convDim, numFilters, numImages))

    # Loop through each filter for each image
    for imageNum in range(numImages):
        for filterNum in range(numFilters):
            convolvedImage = np.zeros(shape=(convDim, convDim))
            filt = np.squeeze(a=W[:, :, filterNum])
            filt = np.rot90(np.squeeze(filt), 2)
            im = np.squeeze(images[:, :, imageNum])
            conI = conv2(im, filt, mode='valid')
            conI = conI + b[filterNum]
            convolvedImage = sigmoid(conI)
            convolvedFeatures[:, :, filterNum, imageNum] = convolvedImage
    return convolvedFeatures

def cnnPool(poolDim, convolvedFeatures):
    """Function to pool the given convolved features

    @param poolDim Dimension of the pooling region
    @param convolvedFeatures convolved features to pool (as given by cnnConvolve)
           convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

    @return Matrix of pooled features in the form
            pooledFeatures(poolRow, poolCol, featureNum, imageNum)
    """
    convolvedDim, _, numFilters, numImages = convolvedFeatures.shape
    pooledFeatures = np.zeros((convolvedDim/poolDim, convolvedDim/poolDim, numFilters, numImages))
    for imageNum in range(numImages):
        for filterNum in range(numFilters):
            pooledImage = np.zeros((convolvedDim/poolDim, convolvedDim,poolDim))
            im = np.squeeze(convolvedFeatures[:, :, filterNum, imageNum])
            polI = conv2(in1=im, in2=np.ones((poolDim, poolDim)), mode='valid')
            #print im.shape, np.ones((poolDim, poolDim)).shape, polI.shape
            polI = polI[::poolDim, ::poolDim]
            #print polI.shape
            polI = polI / float(poolDim**2)
            #print pooledFeatures.shape
            pooledFeatures[:, :, filterNum, imageNum] = polI
    return pooledFeatures

def main():
    # Load the data from the data file
    cur_dir = os.path.dirname(__file__)
    img_dir = os.path.join(cur_dir, '../data/train-images-idx3-ubyte')
    lab_dir = os.path.join(cur_dir, '../data/train-labels-idx1-ubyte')
    timg_dir = os.path.join(cur_dir, '../data/t10k-images-idx3-ubyte')
    tlab_dir = os.path.join(cur_dir, '../data/t10k-labels-idx1-ubyte')
    X, y = loadlocal_mnist(images_path=img_dir, labels_path=lab_dir)
    tX, ty = loadlocal_mnist(images_path=timg_dir, labels_path=tlab_dir)

    # Initialize and input variables
    imageDim = 28
    numImages = X.shape[0]
    try:
        filterDim = int(raw_input("Enter the filter dimensions: "))
        if filterDim<2 or filterDim>imageDim:
            print "Filter dimension must be between 2 and",imageDim,"both inclusive"
            print "Utilizing default as 8\n"
            filterDim = 8
    except Exception as e:
        print "\nInvalid input for the filter dimension. Utilizing default as 8\n"
        filterDim = 8
    try:
        numFilters = int(raw_input("Enter the Number of filters(kernels): "))
    except Exception as e:
        print "\nInvalid input for number of filters(kernels). Utilizing default as 100\n"
        numFilters = 100
    try:
        poolDim = int(raw_input("Enter the pooling dimensions: "))
        if poolDim<2 or poolDim>imageDim-filterDim:
            print "Pooling dimension must be between 2 and",imageDim-filterDim,"both inclusive"
            print "Utilizing default as 3\n"
            poolDim = 3
    except Exception as e:
        print "\nInvalid input for the pooling dimension. Utilizing default as 3\n"
        poolDim = 3

    # Reshape the 1D vector of pixels of each image int a 2D matrix
    X = np.transpose(X)
    X = np.reshape(X, (imageDim, imageDim, numImages), order='F')

    # Initialize the weights
    W = np.random.randn(filterDim, filterDim, numFilters)
    b = np.random.rand(numFilters)
    convImages = X[:, :, :8]

    # Convolve the features
    convolvedFeatures = cnnConvolve(filterDim, numFilters, convImages, W, b)

    # Check the implemented convolution
    for i in range(1000):
        filterNum = randi(0, numFilters-1)
        imageNum = randi(0, 7)
        imageRow = randi(0, imageDim-filterDim)
        imageCol = randi(0, imageDim-filterDim)
        patch = convImages[imageRow:imageRow+filterDim, imageCol:imageCol+filterDim,imageNum]
        feature = sum(sum(patch*W[:, :, filterNum]))+b[filterNum]
        feature = 1.0/(1+np.exp(-feature))

        if abs(feature - convolvedFeatures[imageRow, imageCol, filterNum, imageNum])>1e-9:
            print "Convolved feature doesn't match test feature"
            print "Filter Number: %d" % (filterNum)
            print "Image Number: %d" % (imageNum)
            print "Image Row: %d" % (imageRow)
            print "Image Column: %d" % (imageCol)
            print "Convolved Feature: %.5f" % (convolvedFeatures[imageRow, imageCol, filterNum, imageNum])
            print "Test Feature: %.5f" % (feature)
            return
    print "Congratulations! Your convolution code passed the test."
    #=========================================================================#
    ### PART 2:- pooling
    pooledFeatures = cnnPool(poolDim=poolDim, convolvedFeatures=convolvedFeatures)

    # Check pooling
    testMatrix = np.reshape(np.array(range(1, 65)), (8, 8), order='F')
    expectedMatrix = np.array([[np.mean(testMatrix[:4, :4]), np.mean(testMatrix[:4, 4:])] \
    , [np.mean(testMatrix[4:, :4]), np.mean(testMatrix[4:, 4:])]])
    testMatrix = np.reshape(testMatrix, (8, 8, 1, 1))
    pooledFeatures = np.squeeze(cnnPool(poolDim=4, convolvedFeatures=testMatrix))
    if not np.array_equal(pooledFeatures, expectedMatrix):
        print "Pooling incorrect"
        print "Expected"
        print expectedMatrix
        print "Got"
        print pooledFeatures
        return
    print "Congratulations! Your pooling code passed the test."

if __name__ == '__main__':
    main()
