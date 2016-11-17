from __future__ import division

import numpy as np
import math
from sklearn.model_selection import KFold
import pandas as pd
import time
import csv
import os
import random
import re
from stemming.porter import stem
import sys
import string
import dbSetup
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans

training_data = []
testing_data = []
reviewer_data = []
hotel_data = []
X_train = []
y_train = []
X_test = []
y_test = []

'''
    Helper functions
'''
def undersample(data):
    new_data = []
    check_dict = {}

    for i in range(0, np.shape(data)[0]):
        if data[i][np.shape(data)[1] - 1] == 1.0:
            new_data.append(data[i].tolist())

    break_while = True
    while(break_while):
        random_row_index = random.randint(0, np.shape(data)[0]-1)
        if random_row_index not in check_dict:
            if data[random_row_index][np.shape(data)[1] - 1] == 0.0:
                new_data.append(data[random_row_index])
                check_dict[random_row_index] = 1

                if len(check_dict) == 377:
                    break_while = False

    return np.array(new_data)


def reset_data():
    global X_train
    global y_train
    global X_test

    X_train = import_data(True)
    X_train = undersample(X_train)

    np.random.shuffle(X_train)
    writeExtendedFeaturesTrainTemp(X_train)

    last_col_index = X_train.shape[1] - 1
    y_train = X_train[:, last_col_index]  # Last column in labels
    X_train = np.delete(X_train, -1, 1)  # delete last column of xtrain
    X_train = np.nan_to_num(X_train)
    # X_train = (X_train - X_train.min(0)) / X_train.ptp(0)

    X_test = import_data(False)
    X_test = np.nan_to_num(X_test)
    # X_test = (X_test - X_test.min(0)) / X_test.ptp(0)


def writeExtendedFeatures(train_mode):
    filename = get_file_name(train_mode)
    data = dbSetup.getFeatures(train_mode)
#    np.savetxt(filename, data, delimiter=",")
    with open(filename, 'wb') as f:
        csv.writer(f).writerows(data)


def writeExtendedFeaturesTrainTemp(data):
    filename = get_file_name(True)
    #    np.savetxt(filename, data, delimiter=",")
    with open(filename, 'wb') as f:
        csv.writer(f).writerows(data)


def get_file_name(train_mode):
    if train_mode:
        feature_list_path_type = 'train'
    else:
        feature_list_path_type = 'test'

    feature_list_path = 'challenge_data/yelp_data_' + feature_list_path_type + '_extended_features_Reviews.dat'

    return feature_list_path


def import_data(train_mode):
    data = []
    count = 0

    # filename = get_file_name(train_mode)
    #
    # with open(filename, 'rb') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         if count == 0:
    #             count += 1
    #         else:
    #             if 'None' not in row:
    #                 row.pop(0)  # Get rid of the ID
    #                 dataline = [w.replace('N', '0') for w in row]
    #                 dataline = [w.replace('Y', '1') for w in dataline]
    #                 dataline = map(float, dataline)
    #                 data.append(dataline)
    #
    # return np.array(data)

    return dbSetup.getFeatures(train_mode)


# Parses stop word list and returns an array of stop words
# Stop word list should be a list of words separated by end lines
def read_stop_words():
    f = open('stop.txt', 'r')
    stop_words = f.read()
    stop_words = stop_words.split('\n')
    return stop_words


def write_to_file(filename, data):
    f = open(filename, 'w')
    f.write('Id,Expected\n')
    i = 0
    for item in data:
        f.write('%s' % i)
        f.write(',')
        f.write('%s' % int(item))
        f.write('\n')
        i = i+1
    f.close()


def remove_anomalies(dataset, labels):
    anomaly_indices = np.where(np.array(labels) == 1)[0].tolist()
    return np.delete(dataset, anomaly_indices, 0)


def execute_classifier(use_training, clf):
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
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
    global feature_list_path
    global X_train
    global X_test
    global y_train
    global y_test

    reset_database = False
    reset_parameters = False

    if reset_database:
        dbSetup.setupDatabase()

    dbSetup.initSQLConnection()

    if reset_parameters:
        writeExtendedFeatures(True)
        writeExtendedFeatures(False)

    reset_data()

    num_splits = 3

    kfold = KFold(n_splits=num_splits)

    # Define "classifiers" to be used
    classifiers = {
        # "Kernel Density Estimation": kernel_density,
        #"One Class SVM": one_class_svm,
        # "Local Outlier Factor": local_outlier_factor,
        "Support Vector Classifier": svm.SVC(),
        #"Multi Layer Perceptron": multi_layer_perceptron,
        "Naive Bayes": GaussianNB(),
        #"Decision Tree": decision_tree,
        # "KNN Regressor": knn_regressor,
        "Random Forest": RandomForestClassifier(criterion="entropy", n_estimators=40)
        # "K Means": k_means
    }

    # Load data from dat file

    X_total = X_train
    y_total = y_train

    for name, classifier in classifiers.items():
        accuracy = 0
        for train_index, test_index in kfold.split(X_total):
            # Use indices to seperate out training and test data
            X_train, X_test = X_total[train_index], X_total[test_index]
            y_train, y_test = y_total[train_index], y_total[test_index]

            accuracy += execute_classifier(True, classifier)

        total = accuracy / num_splits
        print "Accuracy of {} is {} %".format(name, round((total)*100, 5))


    reset_data()

    # Use this loop for testing on test data
    for name, classifier in classifiers.items():
        y_test = execute_classifier(False, classifier)
        write_to_file(name + '_output.csv.dat', y_test)

    dbSetup.closeSQLConnection()


if __name__ == "__main__":
    main()
