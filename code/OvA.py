# One vs All
import numpy as np 
import utils 

# error correcting output codes matrix
ecocMat = (np.ones(utils.numOfClasses) * -1) + (np.eye(utils.numOfClasses) * 2)

def main():
    classifiers = []
    data = utils.fetchData()
    divider = int(round(.85 * len(data)))
    trainset = data[:divider]
    validation = data[divider:]

    # train OvA classifiers
    for i in range(utils.numOfClasses):
        binData = utils.transformToBinaryClasses(trainset, positiveClass=i)
        model = utils.SVM(utils.inputDim, utils.eta, 1, 15)
        model.train(binData)
        classifiers.append(model)
        print("finished with #{} model".format(i))

    # validate - Hamming Distance
    correct = 0
    incorrect = 0
    for x, y in validation:
        outputVec, _ = utils.createOutputVectors(x, classifiers)

        y_tag = utils.HammingDistance(ecocMat, outputVec)
        if (y_tag == y):
            correct += 1
        else:
            incorrect += 1
    
    acc = 1. * correct / len(validation)
    print("HD : correct: {} incorrect: {} total: {}\n accuracy: {}".format(\
        correct, incorrect, len(validation), acc))


    # validate - Loss Base Decoding
    correct = 0
    incorrect = 0
    for x, y in validation:
        _, predVec = utils.createOutputVectors(x, classifiers)

        y_tag = utils.lossBaseDecoding(ecocMat, predVec)
        if (y_tag == y):
            correct += 1
        else:
            incorrect += 1
    
    acc = 1. * correct / len(validation)
    print("LBD : correct: {} incorrect: {} total: {}\n accuracy: {}".format(\
        correct, incorrect, len(validation), acc))

    
if __name__ == "__main__":
    main()