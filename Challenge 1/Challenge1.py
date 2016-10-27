from __future__ import division

import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.model_selection import KFold
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

X_Train = []
y_train = []
X_Test = []
y_test = []
'''
    Helper functions
'''
def import_data(file):
    # Loading raw data from the matlab object
    data = (io.loadmat(file))

    # Converting to an array
    dataset = data['x']['data'].tolist()[0][0]
    dataset = np.array(dataset)

    # Getting labels, as a list of lists
    labels = data['x']['nlab'].tolist()[0][0]

    # Flattening them out into one list
    labels = [item for sublist in labels for item in sublist]
    labels = np.array(labels) # 183 are 1 (outliers) and 238 are 2 (inliers)

    return dataset, labels


def remove_anomalies(dataset, labels):
    anomaly_indices = np.where(np.array(labels) == 1)[0].tolist()
    return np.delete(dataset, anomaly_indices, 0)


'''
    Kernel Density Estimation using sci-kit learn
'''
def kernel_density():
    # KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(X_Train)
    score_samples_log = kde.score_samples(X_Test)
    score_samples = np.exp(score_samples_log)

    # plt.plot(score_samples)
    # plt.show()
    accuracy = 0

    for i in range(0, len(score_samples)):
        if score_samples[i] == 0.0 and y_test[i] == 1:
            accuracy += 1
        if score_samples[i] != 0.0 and y_test[i] == 0:
            accuracy += 1

    return accuracy/len(score_samples)


'''
    One Class SVM using sci-kit learn
'''
def one_class_svm():
    total = len(y_train)
    correct = 0

    clf = svm.OneClassSVM(nu=0.9)
    clf.fit(X_Train)
    prediction = clf.predict(X_Test)

    for i in range(0, total):
        if prediction[i] == -1.0 and y_test[i] == 1:
            correct += 1
        if prediction[i] == 1 and y_test[i] == 0:
            correct += 1

    return correct/total


'''
    Implementation of Local Outlier Factor
'''
def local_reachability_distance(nn_indices, distance_data, k):
    rd = 0
    point_index = nn_indices[0]
    neighbour_indices = nn_indices[1:k+1]

    for distance_index in range(0, len(neighbour_indices)):
        true_distance = distance_data[point_index][distance_index + 1]
        k_distance = max(distance_data[point_index])
        rd += max(k_distance, true_distance)
        distance_index += 1

    return 1 / (rd / k)


def local_outlier_factor():
    # find k-nearest neighbours of a point
    lofs = []
    k = 3
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X_Train)
    nn_distances, nn_indices = nbrs.kneighbors(X_Train)

    for index in nn_indices:
        point_index = index[0]
        neighbour_indices = index[1:k+1]

        lrd_sum = 0
        for neighbour_index in neighbour_indices:
            lrd_sum += local_reachability_distance(nn_indices[neighbour_index], nn_distances, k)

        normalized_lrd_n = lrd_sum / k
        lrd_point = local_reachability_distance(nn_indices[point_index], nn_distances, k)
        lofs.append(normalized_lrd_n / lrd_point)

    accuracy = 0
    threshold = 1.2
    for i in range(0, len(lofs)):
        if lofs[i] > threshold and y_test[i] == 1:
            accuracy += 1
        if lofs[i] <= threshold and y_test[i] == 0:
            accuracy += 1

    return accuracy/len(lofs)


'''
    Main function. Start reading the code here
'''
def main():
    global X_Train
    global y_train
    global X_Test
    global y_test

    accuracy = 0

    # Load data from dat file
    X_Train, y_train, X_Test, y_test = import_data('oc_514.mat')

    # Make a kfold object that will split data into k training and test sets

    # Define "classifiers" to be used
    classifiers = {
        "Kernel Density Estimation": kernel_density,
        "One Class SVM": one_class_svm}
        # "Local Outlier Factor": local_outlier_factor}

    for name, classifier in classifiers.items():
        # Every classifier returns an accuracy. We sum and average these for each one
        accuracy = classifier()
        print "Accuracy of {} is {} %".format(name, round((accuracy)*100, 5))


if __name__ == "__main__":
    main()