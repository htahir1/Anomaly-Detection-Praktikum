from __future__ import division

from sklearn import svm
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
import csv

X_Train = []
y_train = []
X_Test = []
y_test = []

'''
    Helper functions
'''
def neighbourhood_average(data):
    '''
        [
            [92	115	120	94	84	102	106	79]
            [84	102	106	79	84	102	102	83]
        ]
    '''

    for i in range(0, np.shape(data)[0]):
        for j in range(0, 4):
            a = j
            sum = 0
            count = 0

            # Getting average of neighbourhood
            for k in range(0, 9):
                if a < np.shape(data)[1]:
                    if data[i][a] != -1:  # This means its NOT a nan value
                        sum += data[i][a]
                        count += 1
                    a += 4

            # Replacing NaNs with average of neighbourhood
            a = j
            for k in range(0, 9):
                if a < np.shape(data)[1]:
                    if data[i][a] == -1:  # This means its a nan value
                        if count != 0:
                            data[i][a] = sum / count
                a += 4


    f = file("preprocessed_training_data.bin", "wb")
    np.save(f, data)
    f.close()

    return data


def neighbourhood_max(data):
    '''
        [
            [92	115	120	94	84	102	106	79]
            [84	102	106	79	84	102	102	83]
        ]
    '''

    for i in range(0, np.shape(data)[0]):
        for j in range(0, 4):
            a = j
            max = -1

            # Getting average of neighbourhood
            for k in range(0, 9):
                if a < np.shape(data)[1]:
                    if data[i][a] != -1:  # This means its NOT a nan value
                        if max < data[i][a]:
                            max = data[i][a]
                    a += 4

            # Replacing NaNs with average of neighbourhood
            a = j
            for k in range(0, 9):
                if a < np.shape(data)[1]:
                    if data[i][a] == -1:  # This means its a nan value
                        data[i][a] = max
                a += 4


    f = file("preprocessed_training_data.bin", "wb")
    np.save(f, data)
    f.close()

    return data


def missing_val_avg(data):
    new_data = []
    # Mask to set or not-set elements in array
    mask = data != -1

    # Compute the mean vaalues for masked elements along each column
    avg = np.true_divide((data * mask).sum(0), mask.sum(0))

    # Finally choose values based on mask and create output dataframe
    new_data = np.where(~mask, avg, data)

    return new_data


def missing_val_max(data):
    new_data = []
    # Mask to set or not-set elements in array
    mask = data != -1

    # Compute the mean vaalues for masked elements along each column
    max = data.max(0)

    # Finally choose values based on mask and create output dataframe
    new_data = np.where(~mask, max, data)

    return new_data


def write_to_file(filename,data):
    f = open(filename, 'w')
    f.write('Id,Expected\n')
    i = 1
    for item in data:
        f.write('%s' % i)
        f.write(',')
        f.write('%s' % item)
        f.write('\n')
        i = i+1
    f.close()


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


def remove_anomalies(dataset, labels):
    anomaly_indices = np.where(np.array(labels) == 1)[0].tolist()
    return np.delete(dataset, anomaly_indices, 0)


'''
    Kernel Density Estimation using sci-kit learn
'''
def kernel_density(use_training):
    # KDE
    y_test = []

    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_Train)
    score_samples_log = kde.score_samples(X_Test)
    score_samples = np.exp(score_samples_log)

    # plt.plot(score_samples)
    # plt.show()

    for i in range(0, len(X_Test)):
        if score_samples[i] == 0.0:
            y_test.append(1)
        else:
            y_test.append(0)

    return y_test

'''
    One Class SVM using sci-kit learn
'''
def one_class_svm(use_training):
    y_test = []

    clf = svm.OneClassSVM(nu=0.01)
    clf.fit(X_Train)
    prediction = clf.predict(X_Test)

    for i in range(0, len(X_Test)):
        if prediction[i] == -1.0:
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


def local_outlier_factor(use_training):
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
        else:
            y_test.append(0)


'''
    SVC
'''
def support_vector(use_training):

    classifier = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                         decision_function_shape='ovo', degree=3, gamma='auto', kernel='poly',
                         max_iter=-1, probability=False, random_state=None, shrinking=True,
                         tol=0.001, verbose=False)

    classifier.fit(X_Train, y_train)
    predictions = classifier.predict(X_Test)

    if not use_training:
        return predictions
    else:
        accuracy = 0
        for i in range(0, len(predictions)):
            if predictions[i] == y_test[i]:
                accuracy += 1

        return accuracy/len(predictions)


def multi_layer_perceptron(use_training):
    clf = MLPClassifier(activation='relu', alpha=1e-7, batch_size='auto',
                        beta_1=0.9, beta_2=0.999, early_stopping=False,
                        epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='adaptive',
                        learning_rate_init=0.001, max_iter=10000, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=1000, shuffle=True,
                        solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                        warm_start=False)

    clf.fit(X_Train, y_train)

    predictions = clf.predict(X_Test)

    if not use_training:
        return predictions
    else:
        accuracy = 0
        for i in range(0, len(predictions)):
            if predictions[i] == y_test[i]:
                accuracy += 1

        return accuracy/len(predictions)

'''
    Main function. Start reading the code here
'''
def main():
    global X_Train
    global y_train
    global X_Test
    global y_test

    # Make a kfold object that will split data into k training and test sets
    num_splits = 3
    kfold = KFold(n_splits=num_splits)

    # Define "classifiers" to be used
    classifiers = {
        # "Kernel Density Estimation": kernel_density,
        # "One Class SVM": one_class_svm}
        # "Local Outlier Factor": local_outlier_factor,
        "Support Vector Classifier": support_vector,
        "Multi Layer Perceptron": multi_layer_perceptron
    }

    # Load data from dat file
    X_Total, y_total, X_Test = import_data('sat-test-data.csv.dat', 'sat-train.csv.dat')
    X_Train = neighbourhood_max(X_Total)

    # Use this loop for testing on training data
    for name, classifier in classifiers.items():
        accuracy = 0
        for train_index, test_index in kfold.split(X_Total):
            # Use indices to seperate out training and test data
            X_Train, X_Test = X_Total[train_index], X_Total[test_index]
            y_train, y_test = y_total[train_index], y_total[test_index]

            accuracy += classifier(use_training=True)

        total = accuracy / num_splits
        print "Accuracy of {} is {} %".format(name, round((total)*100, 5))


    # Load the data
    X_Train, y_train, X_Test = import_data('sat-test-data.csv.dat', 'sat-train.csv.dat')
    X_Train = neighbourhood_max(X_Train)
    X_Test = neighbourhood_max(X_Test)

    # Use this loop for testing on test data
    for name, classifier in classifiers.items():
        y_test = classifier(use_training=False)
        write_to_file(name + '_output.csv.dat', y_test)


if __name__ == "__main__":
    main()