# One vs All
import numpy as np 
import utils 

# error correcting output codes matrix
ecocMat = (np.ones(utils.numOfClasses) * -1) + (np.eye(utils.numOfClasses) * 2)

def main():
    classifiers = []
    trainset = utils.fetchData()
    testset = utils.loadTestData()

    # train OvA classifiers
    for i in range(utils.numOfClasses):
        binData = utils.transformToBinaryClasses(trainset, positiveClass=i)
        model = utils.SVM(utils.inputDim, utils.eta, 1, 15)
        model.train(binData)
        classifiers.append(model)
        print("finished with #{} model".format(i))

    # evaluate test data by Hamming Distance
    utils.evaluation(testset, utils.HammingDistance,\
        ecocMat, 'test.onevall.ham.pred', \
        classifiers, distanceMetric="Hamming")

    # evaluate test data by Loss Base Decoding
    utils.evaluation(testset, utils.lossBaseDecoding,\
        ecocMat, 'test.onevall.loss.pred', \
        classifiers, distanceMetric="LBD")

if __name__ == "__main__":
    main()