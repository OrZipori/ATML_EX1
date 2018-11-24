import numpy as np
import itertools
import random
import utils


def get_all_pairs(classes):
    return itertools.combinations(classes, 2)


def filter_data(data, classes_to_filter):
    filtered_data = []
    for x, y in data:
        if classes_to_filter.__contains__(y):
            filtered_data.append((x, y))

    return filtered_data


def update_ecoc_matrix(ecoc_matrix, classifier_index, y0, y1, factor=0.4):
    ecoc_matrix[y0, classifier_index] = 1
    # y1 is now a pair
    ecoc_matrix[y1[0], classifier_index] = -1
    ecoc_matrix[y1[1], classifier_index] = -1
    for i in range(ecoc_matrix.shape[0]):
        if i == y0 or i == y1[0] or i == y1[1]:
            continue

        factor = factor * random.randint(-1, 1)
        ecoc_matrix[i, classifier_index] = factor


def main():
    k = utils.numOfClasses
    columns = int(((k * k) * (k - 1)) / 4) 
    ecoc_matrix = np.zeros((k, columns), dtype=float)
    classifiers = []
    trainset = utils.fetchData()
    print("total train set data len: {}".format(str(len(trainset))))
    testset = utils.loadTestData()

    lambda_p = 1
    epoch_number = 15
    pair_index = 0

    # Train All-Pair Classifiers
    for i in range(utils.numOfClasses):
        # add all other classes that are not the positive class
        oppsiteClasses = [c for c in range(utils.numOfClasses) if c != i]
        for y0, y1 in get_all_pairs(oppsiteClasses):
            update_ecoc_matrix(ecoc_matrix, pair_index, i, (y0, y1))
            print("working on {} vs {},{}".format(i, y0, y1))
            pair_index = pair_index + 1
            filtered_data = filter_data(trainset, (i, y0, y1))
            print("relevant data: {}".format(str(len(filtered_data))))
            binary_data = utils.transformToBinaryClasses(filtered_data, positiveClass=i)
            model = utils.SVM(utils.inputDim, utils.eta, lambda_p, epoch_number)
            model.train(binary_data)
            classifiers.append(model)
            print("finished with #{} model".format(pair_index))

    print(ecoc_matrix)

    # Evaluate Test Data by Hamming Distance
    utils.evaluation(testset
                     , utils.HammingDistance
                     , ecoc_matrix
                     , 'test.random.ham.pred'
                     , classifiers
                     , distanceMetric="Hamming")

    # Evaluate Test Data by Loss Base Decoding
    utils.evaluation(testset
                     , utils.lossBaseDecoding
                     , ecoc_matrix
                     , 'test.random.loss.pred'
                     , classifiers
                     , distanceMetric="LBD")


if __name__ == "__main__":
    main() 