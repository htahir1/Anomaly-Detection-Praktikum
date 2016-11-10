from __future__ import division

import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import time
import csv

training_data = []
testing_data = []
reviewer_data = []
hotel_data = []

'''
    Helper functions
'''
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


def import_data():
    training_data = []
    testing_data = []
    reviewer_data = []
    hotel_data = []

    hotel_file_name = 'challenge_data/yelp_data_hotel.dat'
    reviewer_file_name =  'challenge_data/yelp_data_reviewer.dat'
    test_file_name =  'challenge_data/yelp_data_test.dat'
    train_file_name =  'challenge_data/yelp_data_train.dat'

    testing_data = pd.read_csv(train_file_name, sep=';', error_bad_lines=False)

    training_data = pd.read_csv(test_file_name, sep=';', error_bad_lines=False)

    reviewer_data = pd.read_csv(reviewer_file_name, sep=';', error_bad_lines=False)

    hotel_data = pd.read_csv(hotel_file_name, sep=';', error_bad_lines=False)

    return training_data, testing_data, reviewer_data, hotel_data


def remove_anomalies(dataset, labels):
    anomaly_indices = np.where(np.array(labels) == 1)[0].tolist()
    return np.delete(dataset, anomaly_indices, 0)


def execute_classifier(use_training, clf):
    clf.fit(X_Train, y_train)

    predictions = clf.predict(X_Test)
    predictions = np.round(predictions)

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
    global training_data
    global testing_data
    global reviewer_data
    global hotel_data

    # Make a kfold object that will split data into k training and test sets
    num_splits = 6
    kfold = KFold(n_splits=num_splits)

    # Define "classifiers" to be used
    classifiers = {
        # "Kernel Density Estimation": kernel_density,
        #"One Class SVM": one_class_svm,
        # "Local Outlier Factor": local_outlier_factor,
        # "Support Vector Classifier": support_vector,
        #"Multi Layer Perceptron": multi_layer_perceptron,
        #"Naive Bayes": naive_bayes,
        #"Decision Tree": decision_tree,
        # "KNN Regressor": knn_regressor,
        # "Random Forest": random_forest
        # "K Means": k_means
    }

    # Load data from dat file
    training_data, testing_data, reviewer_data, hotel_data = import_data()

    # Use this loop for testing on training data
    # for name, classifier in classifiers.items():
    #     accuracy = 0
    #     for train_index, test_index in kfold.split(X_Total):
    #         # Use indices to seperate out training and test data
    #         X_Train, X_Test = X_Total[train_index], X_Total[test_index]
    #         y_train, y_test = y_total[train_index], y_total[test_index]
    #
    #         accuracy += execute_classifier(True, classifier)
    #
    #     total = accuracy / num_splits
    #     print "Accuracy of {} is {} %".format(name, round((total)*100, 5))


    # Load the data
    training_data, testing_data, reviewer_data, hotel_data = import_data()

    # Use this loop for testing on test data
    for name, classifier in classifiers.items():
        y_test = execute_classifier(True, classifier)
        write_to_file(name + '_output.csv.dat', y_test)


if __name__ == "__main__":
    main()