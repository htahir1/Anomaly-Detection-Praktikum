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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


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


def normalize_data(x):
    return x / x.max(axis=0)


def reset_data():
    global X_train
    global y_train
    global X_test

    X_train = dbSetup.getFeatures(train_mode=True)
    X_test = dbSetup.getFeatures(train_mode=False)


    X_train = deleteColumnsPanda(X_train,['id', 'proto','service','state', 'attack_cat'])
    X_test = deleteColumnsPanda(X_test,['id', 'proto', 'service', 'state'])

    X_train = X_train.as_matrix()
    X_test = X_test.as_matrix()

    pca = PCA(n_components = 30)
    pca.fit(X_train)

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    np.random.shuffle(X_train)

    last_col_index = X_train.shape[1] - 1
    y_train = X_train[:, last_col_index]  # Last column in labels
    X_train = np.delete(X_train, -1, 1)  # delete last column of xtrain

    # X_train = normalize_data(X_train)
    # X_test = normalize_data(X_test)


def get_file_name(train_mode):
    if train_mode:
        feature_list_path_type = 'train'
    else:
        feature_list_path_type = 'test'

    feature_list_path = 'challenge_data/yelp_data_' + feature_list_path_type + '_extended_features_Reviews.dat'

    return feature_list_path


def import_data(train_mode):
    return dbSetup.getFeatures(train_mode)


def write_predictions_to_file(filename, data):
    f = open(filename, 'w')
    f.write('id,label\n')
    i = 1
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

def deleteColumnsPanda(pandaDataframe, blacklist):
    for element in blacklist:
        del pandaDataframe[element]
    return pandaDataframe


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

    if reset_database:
        dbSetup.setupDatabase()

    dbSetup.initSQLConnection()
    #
    # Features = list()
    # X_train, X_test, Features  = dbSetup.import_data();
    # X_train_cleaned = deleteColumnsPanda(X_train,['proto','service','state','attack_cat','label'])
    # clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    # clf.fit(X_train_cleaned)
    # y_pred_train = clf.predict(X_train_cleaned)
    # print y_pred_train #predicted on X_train & not cleaned!

    num_splits = 3

    kfold = KFold(n_splits=num_splits)

    # Define "classifiers" to be used
    classifiers = {
        # "Support Vector Classifier": svm.SVC(),
        # "Multi Layer Perceptron": multi_layer_perceptron,
        # "Naive Bayes": GaussianNB(),
        # "Random Forest": RandomForestClassifier(criterion="entropy", n_estimators=40),
        # "Kmeans": KMeans(n_clusters=11, random_state=0),
        "KNN": KNeighborsClassifier(n_neighbors=3)
        # "K Means": k_means
    }

    # Load data from dat file
    reset_data()
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
        write_predictions_to_file(name + '_output.csv.dat', y_test)

    dbSetup.closeSQLConnection()


if __name__ == "__main__":
    main()
