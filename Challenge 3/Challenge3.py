from __future__ import division

from sklearn import manifold
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
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

attack_cats = []
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
def undersample(data, how_many_extra_normal=0):
    new_data = []
    check_dict = {}

    np.random.shuffle(data)

    for i in range(0, np.shape(data)[0]):
        if data[i][np.shape(data)[1] - 1] == 1:
            new_data.append(data[i].tolist())

    size_of_anomalies = len(new_data)
    
    break_while = True
    while(break_while):
        random_row_index = random.randint(0, np.shape(data)[0]-1)
        if random_row_index not in check_dict:
            if data[random_row_index][np.shape(data)[1] - 1] == 0:
                new_data.append(data[random_row_index])
                check_dict[random_row_index] = 1

                if len(check_dict) == size_of_anomalies + how_many_extra_normal:
                    break_while = False

    return np.array(new_data)


def normalize_data(x):
    return (x - x.min(0)) / x.ptp(0)


def reset_data(with_undersampling=True):
    global X_train
    global y_train
    global X_test
    global attack_cats

    X_train = dbSetup.getFeatures(train_mode=True)
    X_test = dbSetup.getFeatures(train_mode=False)

    X_train = deleteColumnsPanda(X_train, ['id', 'proto','service','state', 'attack_cat'])
    X_test = deleteColumnsPanda(X_test, ['id', 'proto', 'service', 'state'])
    X_train = X_train.as_matrix()
    X_test = X_test.as_matrix()

    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    # X_train = np.nan_to_num(X_train)
    # X_test = np.nan_to_num(X_test)

    if with_undersampling:
        X_train = undersample(X_train)

    np.random.shuffle(X_train)

    last_col_index = X_train.shape[1] - 1
    y_train = X_train[:, last_col_index].astype(int)  # Last column in labels
    X_train = np.delete(X_train, -1, 1)  # delete last column of xtrain

    last_col_index = X_train.shape[1] - 1
    attack_cats = X_train[:, last_col_index]  # Last column in labels
    #X_train = np.delete(X_train, -1, 1)  # delete last column of xtrain



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


def visualize_anomalies():
    data = dbSetup.getFeatures(train_mode=True)
    data = deleteColumnsPanda(data,['id', 'proto','service','state'])

    data = data.as_matrix()

    data = undersample(data, how_many_extra_normal=40)

    second_last_col_index = data.shape[1] - 2
    labels = data[:, second_last_col_index]
    data = np.delete(data, -1, 1)  # delete last column of xtrain
    data = np.delete(data, -1, 1)  # delete last column of xtrain

    tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    Y = tsne.fit_transform(data)

    plt.scatter(
        Y[:, 0], Y[:, 1], marker='o',
        cmap=plt.get_cmap('Spectral'))
    for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.show()


def local_reachability_distance(nn_indices, distance_data, k):
    rd = 0
    point_index = nn_indices[0]
    neighbour_indices = nn_indices[1:k+1]
    #
    # for distance_index in range(0, len(neighbour_indices)):
    #     true_distance = distance_data[point_index][distance_index + 1]  # Find actual distance
    #     k_distance = max(distance_data[point_index]) #
    #     rd += max(k_distance, true_distance)
    #     distance_index += 1

    return 1 / (max(distance_data[point_index]) / k)


def unsupervised_technique(X_train, y_train, X_test, y_test, attack_cats_train):
    # find k-nearest neighbours of a point
    lofs = []
    k = 3
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X_train)  # Find nearest neighbours
    nn_distances, nn_indices = nbrs.kneighbors(X_train)

    for index in nn_indices:  # For all points
        point_index = index[0]  # The first one is the point itself

        neighbour_indices = index[1:k+1]  # The rest are the nearest neighbors

        print "**************************"
        print attack_cats_train[point_index]
        for i in range(0, len(neighbour_indices)):
            print attack_cats_train[neighbour_indices[i]], nn_distances[point_index][i+1]


    #     plt.show()
    #
    #     lrd_sum = 0
    #     for neighbour_index in neighbour_indices:  # All neighbors of neighbors
    #         lrd_sum += local_reachability_distance(nn_indices[neighbour_index], nn_distances, k)
    #
    #     normalized_lrd_n = lrd_sum / k  # Take average of LRD's of all neighbor of neighbors
    #     lrd_point = local_reachability_distance(nn_indices[point_index], nn_distances, k)
    #     lofs.append(normalized_lrd_n / lrd_point)
    #
    # accuracy = 0
    # threshold = 1.2
    # for i in range(0, len(lofs)):
    #     if lofs[i] > threshold and y_train[i] == 0:
    #         accuracy += 1
    #     if lofs[i] <= threshold and y_train[i] == 1:
    #         accuracy += 1

    # return accuracy/len(lofs)

def kmeans():
    print "\n****************** K MEANS ****************************\n"
    clf = KMeans(n_clusters=2)
    clf.fit(X_train)

    predictions = clf.predict(X_test)

    for i in range(0, len(attack_cats)):
        print predictions[i], y_train[i], attack_cats[i]
'''
    Main function. Start reading the code here
'''
def main():
    global training_data
    global testing_data
    global X_train
    global X_test
    global y_train
    global y_test
    global attack_cats

    reset_database = False

    if reset_database:
        dbSetup.setupDatabase()

    dbSetup.initSQLConnection()

    #visualize_anomalies()

    num_splits = 3

    kfold = KFold(n_splits=num_splits)

    # Define "classifiers" to be used
    classifiers = {
        # "Support Vector Classifier": svm.SVC(),
        # "Multi Layer Perceptron": multi_layer_perceptron,
        # "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(criterion="entropy", n_estimators=40),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=3)
         #"K Means": k_means
    }

    # Load data from dat file
    reset_data(with_undersampling=True)
    unsupervised_technique(X_train, y_train, X_test, y_test, attack_cats)

    #reset_data(with_undersampling=True)
    #kmeans()

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
