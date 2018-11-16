import numpy as np 
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

# global variables
eta = 0.1

def fetchData():
    # read data
    mnist = fetch_mldata("MNIST original", data_home="./data")
    
    X, Y = mnist.data[:60000] / 255., mnist.target[:60000]
    x = [np.reshape(ex, (1, 784)) for ex, ey in zip(X, Y) if ey in [0, 1, 2, 3]]
    y = [ey for ey in Y if ey in [0, 1, 2, 3]]
    # suffle examples
    x, y = shuffle(x, y, random_state=1)

    return list(zip(x, y))

'''
 since we have multiclass data but wish to train binray classifier
 we need to have a function that transforms classes to binray classification
 parameter: data is a list of pairs of x, y where y had value only from 2 classes
            positiveClass is the class that will be mapped to +1 all other will be
                          mapped to -1
'''
def transformToBinaryClasses(data, positiveClass):
    newData = []
    for x, y in data:
        if (y == positiveClass):
            newData.append((x, 1))
        else:
            newData.append((x, -1))

    return newData

class SVM(object):
    def __init__(self, inputDim, eta, lambdaP, epochs):
        self.W = np.zeros((inputDim, 1)) # weights
        self.eta = eta # learning rate
        self.lambdaP = lambdaP # regularization parameter
        self.epochs = epochs
    
    def train(self, trainset):
        # iterate over epochs
        for t in range(1, self.epochs + 1):
            np.random.shuffle(trainset)
            # we want to diminish eta the more we progress, to make "small steps", no to loose min
            eta = self.eta / np.sqrt(t)
            loss = 0

            # iterate over trainset
            for x, y in trainset:
                pred = y * (np.dot(x, self.W)[0][0]) # get the scalar
                reg = (eta * self.lambdaP) * self.W # regularization
                if (1 - pred >= 0): # check hinge loss
                    lossAddition = (eta * y * x)
                    # need to reshape since x and W are not the same shape
                    self.W += np.reshape(lossAddition, (784, 1)) - reg 
                else:
                    self.W -= reg
                loss += 1 - pred
            
            print("Loss for #{} epoch is {}".format(t, loss))

    def inference(self, x):
        # since we use binary classifier
        return np.sign(np.dot(x, self.W))[0][0]

    