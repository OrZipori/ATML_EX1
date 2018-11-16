# Test SVM model
import utils
import numpy as np 

def main():
    data = [(x, y) for x,y in utils.fetchData() if y in [0, 2]]
    binaryData = utils.transformToBinaryClasses(data, positiveClass=0)

    divider = int(round(.85 * len(binaryData)))
    validation = binaryData[divider:]
    trainset = binaryData[:divider]
    inputDim = 784

    model = utils.SVM(inputDim, utils.eta, 1, 10)
    model.train(trainset, printLoss=True)

    # validate
    correct = 0 
    incorrect = 0
    for x, y in validation:
        y_tag = model.inference(x)
        if (y == y_tag):
            correct +=1
        else:
            incorrect +=1

    acc = 1. * correct / len(validation)
    print("correct: {} incorrect: {} total: {} \n accuracy: {}".format(\
        correct, incorrect, len(validation), acc))
    

if __name__ == "__main__":
    main()
