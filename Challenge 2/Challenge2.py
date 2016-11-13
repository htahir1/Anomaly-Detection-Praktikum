from __future__ import division

import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import time
import csv
import os
import re
from stemming.porter import stem
import sys
import math
import string
import dbSetup

training_data = []
testing_data = []
reviewer_data = []
hotel_data = []
feature_list_path_type = 'train'
feature_list_path = 'challenge_data/yelp_data_' + feature_list_path_type + '_extended_features_Reviews.dat'

'''
    Helper functions
'''
def writeExtendedFeatures(header, data, filename):
    with open(filename, 'w') as file:
        for i in range(0,len(header)-1):
            file.write(str(header[i]))
            file.write(",")
        file.write(str(header[len(header)-1])+"\n")

        for row in data:
            for i in range(0,len(row)-1):
                file.write(str(row[i])+",")
            file.write("%s\n" % str(row[len(row)-1]))


# Takes a string and sees if there's a number in it
def has_number(word):
    for i in word:
        if i >= '0' and i <= '9':
            return False
    return True


# Takes a string and checks if its a number
def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


# Parses stop word list and returns an array of stop words
# Stop word list should be a list of words separated by end lines
def read_stop_words():
    f = open('stop.txt', 'r')
    stop_words = f.read()
    stop_words = stop_words.split('\n')
    return stop_words


# Removes punctuation from a string
def remove_punctuation(s):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    return s.translate(replace_punctuation)


# Takes a word and normalizes it to a type
def normalize_word(word):
    return stem(word)


# Takes some data and removes punctuation and whitespace
def clean_data(some_data):
    some_data = remove_punctuation(some_data)
    some_data = re.sub("[\t\n]", ' ', some_data)
    return some_data


# Breaks corpus down into an inverted index
def make_index_1():
    # traverse root directory, and list directories as dirs and files as files
    global corpus_path
    global index_terms
    global index_tokens
    token_count = 0
    stop_word_list = read_stop_words()
    for root, dirs, files in os.walk(corpus_path):
        for file_id in files:
            folder = os.path.basename(root) + '/'
            key = folder + file_id
            f = open(corpus_path + key, 'r')
            file_data = f.read()
            file_data = clean_data(file_data)
            file_data = file_data.split(' ')  # split file_data

            for word in file_data:  # makes the inverted index
                if len(word) > 1:
                    if word != 'html' and word != 'pre':
                            token_count += 1
                            try:
                                if word in index_tokens:
                                    index_tokens[word] += 1
                                else:
                                    index_tokens[word] = 1
                            except:
                                print word

                            word = word.lower()
                            if word not in stop_word_list:
                                word = normalize_word(word)
                                if word in index_terms:
                                    index_terms[word] += 1
                                else:
                                    index_terms[word] = 1

    return token_count


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
    global feature_list_path

    reset_database = False
    reset_parameters = True
    train_mode = True

    if reset_database:
        dbSetup.setupDatabase()

    if reset_parameters:
        dbSetup.initSQLConnection()
        data, header = dbSetup.getFeaturesByReview(train_mode)
        writeExtendedFeatures(header, data, feature_list_path)
        print (header)


    #write_to_file('yelp_data_test_review_length.csv.dat', ret)
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

    # Use this loop for testing on test data
    for name, classifier in classifiers.items():
        y_test = execute_classifier(True, classifier)
        write_to_file(name + '_output.csv.dat', y_test)


if __name__ == "__main__":
    main()
