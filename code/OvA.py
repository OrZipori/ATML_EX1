# One vs All
import numpy as np 
import utils 

# error correcting output codes matrix
ecocMat = (np.ones(utils.numOfClasses) * -1) + (np.eye(utils.numOfClasses) * 2)

def main():
    classifiers = []
    data = utils.fetchData()

    # train OvA classifiers
    for i in range(utils.numOfClasses):
        binData = utils.transformToBinaryClasses(data, positiveClass=i)
        model = utils.SVM(utils.inputDim, utils.eta, 1, 10)
        model.train(binData)
        classifiers.append(model)
    
if __name__ == "__main__":
    main()