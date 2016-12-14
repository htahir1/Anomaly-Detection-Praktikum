from __future__ import division

from sklearn.model_selection import KFold
import random
import json
import dbSetup
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition

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


def get_data(training):
    if training:
        filename = "data/training_set_dedup"
    else:
        filename = "data/test_set_dedup"

    data = []
    with open(filename) as data_file:
        for row in data_file:
            j = json.loads(row)
            data.append(j)

    return data


def normalize_data(x):
    return normalize(x, norm='l2')


def reset_data(with_undersampling=True):
    global X_train
    global y_train
    global X_test

    X_train = get_data(training=True)
    X_test = get_data(training=False)

    # X_train = deleteColumnsPanda(X_train, ['id', 'proto','service','state'])
    # X_test = deleteColumnsPanda(X_test, ['id', 'proto', 'service', 'state'])
    # X_train = X_train.as_matrix()
    # X_test = X_test.as_matrix()
    #
    # # X_train = np.nan_to_num(X_train)
    # # X_test = np.nan_to_num(X_test)
    #
    # if with_undersampling:
    #     X_train = undersample(X_train)
    #
    # np.random.shuffle(X_train)
    #
    # last_col_index = X_train.shape[1] - 1
    # y_train = X_train[:, last_col_index].astype(int)  # Last column in labels
    # X_train = np.delete(X_train, -1, 1)  # delete last column of xtrain
    #
    # X_train = normalize_data(X_train)
    # X_test = normalize_data(X_test)


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

    data = undersample(data, how_many_extra_normal=-40)

    second_last_col_index = data.shape[1] - 2
    labels = data[:, second_last_col_index]
    data = np.delete(data, -1, 1)  # delete last column of xtrain
    data = np.delete(data, -1, 1)  # delete last column of xtrain

    # tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    # Y = tsne.fit_transform(data)
    pca = decomposition.PCA(n_components=2)
    pca.fit(data)
    Y = pca.transform(data)

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

    reset_database = False

    if reset_database:
        dbSetup.setupDatabase()

    # visualize_anomalies()

    num_splits = 3

    kfold = KFold(n_splits=num_splits)

    # Define "classifiers" to be used
    classifiers = {
        "Random Forest": RandomForestClassifier(criterion="entropy", n_estimators=40),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=3)
    }

    # Load data from dat file
    # unsupervised_technique(X_train, y_train, X_test, y_test, attack_cats)


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


if __name__ == "__main__":
    main()
