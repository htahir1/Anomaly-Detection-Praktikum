from __future__ import division

from sklearn import svm
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.neighbors import NearestNeighbors
import csv

X_Train = []
y_train = []
X_Test = []

'''
    Helper functions
'''
def parse_file(filename):
    data = []

    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            dataline = [w.replace('NaN', '-1') for w in row]  # Convert NaN to -1
            dataline = map(int, dataline)
            data.append(dataline)

    return np.array(data)


def import_data(test_file_name, train_file_name):
    xtrain = []
    ytrain = []
    xtest = []

    xtrain = parse_file(train_file_name)
    last_col_index = xtrain.shape[1]-1
    ytrain = xtrain[:, last_col_index]  # Last column in labels
    xtrain = np.delete(xtrain, -1, 1)  # delete last column of xtrain

    xtest = parse_file(test_file_name)

    return xtrain, ytrain, xtest

# data = (io.loadmat(file))
    #
    # # Converting to an array
    # dataset = data['x']['data'].tolist()[0][0]
    # dataset = np.array(dataset)
    #
    # # Getting labels, as a list of lists
    # labels = data['x']['nlab'].tolist()[0][0]
    #
    # # Flattening them out into one list
    # labels = [item for sublist in labels for item in sublist]
    # labels = np.array(labels) # 183 are 1 (outliers) and 238 are 2 (inliers)
    #
    # return dataset, labels


def remove_anomalies(dataset, labels):
    anomaly_indices = np.where(np.array(labels) == 1)[0].tolist()
    return np.delete(dataset, anomaly_indices, 0)


'''
    Kernel Density Estimation using sci-kit learn
'''
def kernel_density():
    # KDE
    y_test = []

    kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(X_Train)
    score_samples_log = kde.score_samples(X_Test)
    score_samples = np.exp(score_samples_log)

    # plt.plot(score_samples)
    # plt.show()

    for i in range(0, len(X_Test)):
        if score_samples[i] == 0.0:
            y_test.append(1)
        elif score_samples[i] != 0.0:
            y_test.append(1)
        else:
            y_test.append(0)

    return y_test

'''
    One Class SVM using sci-kit learn
'''
def one_class_svm():
    y_test = []

    clf = svm.OneClassSVM(nu=0.9)
    clf.fit(X_Train)
    prediction = clf.predict(X_Test)

    for i in range(0, len(X_Test)):
        if prediction[i] == -1.0:
            y_test.append(1)
        elif prediction[i] == 1:
            y_test.append(1)
        else:
            y_test.append(0)

    return y_test

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
    y_test = []
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

    threshold = 1.2
    for i in range(0, len(X_Test)):
        if lofs[i] > threshold:
            y_test.append(1)
        elif lofs[i] <= threshold:
            y_test.append(1)
        else:
            y_test.append(0)


'''
    Main function. Start reading the code here
'''
def main():
    global X_Train
    global y_train
    global X_Test

    # Load data from dat file
    X_Train, y_train, X_Test = import_data('sat-test-data.csv.dat', 'sat-train.csv.dat')

    # Make a kfold object that will split data into k training and test sets

    # Define "classifiers" to be used
    classifiers = {
        "Kernel Density Estimation": kernel_density,
        "One Class SVM": one_class_svm}
        # "Local Outlier Factor": local_outlier_factor}

    for name, classifier in classifiers.items():
        # Every classifier returns an accuracy. We sum and average these for each one
        y_test = classifier()
        print y_test

if __name__ == "__main__":
    main()