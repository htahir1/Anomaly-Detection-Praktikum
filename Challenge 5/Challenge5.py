from __future__ import division

import random
import json
import dbSetup
from ObjDumpHandler import ObjDumpHandler
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import csv
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster

X_train = []
y_train = []
X_test = []
y_test = []
DLLs = []
feature_names = []
# feature_names.append("Size of sample")     # 0
feature_names.append("# of Exports")       # 1
feature_names.append("# of Imports")       # 2
feature_names.append("PEsecs(Size)")       # 3
feature_names.append("PESecs(VirtSize)")   # 4
feature_names.append("PESecs(Entropy)")    # 5
# feature_names.append("pehash")             # 6
feature_names.append("debug")              # 7
feature_names.append("rich_header")        # 8


'''
    Helper functions
'''

def get_filename(training):
    if training:
        filename = "data/training_set_dedup"
    else:
        filename = "data/test_set_dedup"

    return filename


def get_data(training):

    filename = get_filename(training)
    data = []
    with open(filename) as data_file:
        for row in data_file:
            j = json.loads(row)
            data.append(j)

    return data


def normalize_data(x,mode):
    if mode == 'l2':
        return normalize(x, norm='l2')
    else:
        return ((x - np.amin(x, axis = 0)) / x.ptp(0))





def process_train_test():
    global X_train
    global y_train
    global X_test

    #np.random.shuffle(X_train)

    last_col_index = X_train.shape[1] - 1
    y_train = X_train[:, last_col_index]  # Last column in labels
    X_train = np.delete(X_train, -1, 1)  # delete last column of xtrain

    X_train = normalize_data(X_train, "l22")
    X_test = normalize_data(X_test, "l22")

    #scaler = StandardScaler()
     # Don't cheat - fit only on training data
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
     # apply same transformation to test data
    #X_test = scaler.transform(X_test)


def write_extended_features():
    global X_train
    global y_train
    global X_test
    global DLLs
    X_train = get_data(training=True)
    X_test = get_data(training=False)
    DLLs = get_import_dlls(X_train)
    X_train = process_data(X_train)
    X_test = process_data(X_test)

    # Binary data
    np.save('data/training_extended_binary.npy', X_train)
    np.save('data/testing_extended_binary.npy', X_test)
    np.savetxt("data/training_extended.csv", np.asarray(X_train), delimiter=",",fmt='%.2f')
    np.savetxt("data/testing_extended.csv", np.asarray(X_test), delimiter=",",fmt='%.2f')

def remove_test_benign():
    Tmp = np.delete(X_test, Test_Benign_list, axis=0)
    return Tmp


def reset_data(with_undersampling=True, reset_extended=True,remove_benign = False):
    global X_train
    global y_train
    global X_test

    if reset_extended:
        write_extended_features()
    else:
        X_train = np.load("data/training_extended_binary.npy")
        X_test = np.load("data/testing_extended_binary.npy")

    if remove_benign:
        X_train = X_train[X_train[:,(X_train.shape[1] - 1)] == 1]
        X_test = remove_test_benign()
        print X_test.shape
    process_train_test()


def import_data(train_mode):
    return dbSetup.getFeatures(train_mode)


def write_predictions_to_file(filename, data):
    f = open(filename, 'w')
    f.write('sha256,label\n')
    i = 0
    filename = get_filename(False)
    with open(filename) as data_file:
        for row in data_file:
            j = json.loads(row)
            f.write('%s' % j["sha256"])
            f.write(',')
            f.write('%s' % 'benign' if data[i] == 0 else 'malicious')
            f.write('\n')

            i = i + 1

    f.close()


def execute_classifier(use_training, clf, name, feature_importance=False):
    global feature_names

    #print "Beginning evaluation of: " + name
    clf.warm_start = True;
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    predictions = np.round(predictions)

    if feature_importance:
        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X_train.shape[1]):
            print("%d. feature %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

        test = []
        for f in range(X_train.shape[1]):
            test.append(indices[f])

        print indices
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances (" + name + ")")
        plt.bar(range(X_train.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(X_train.shape[1]), indices)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()

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


def generic_numeric(clf_name,clf,instance,min,max,element):
    classifier = {}
    for i in range(min,max):
        tmp = clf()
        tmp.__setattr__(element,i)
        classifier["_".join((clf_name,element,str(i)))] = tmp
    return classifier

def cluster_data(mode_is_training,clf):
    clf.warm_start = True
    if mode_is_training:
        clf.fit(X_total)
        return clf.labels_

    else:
        clf.fit(X_total)
        print "Returning Test Clusters"
        return clf.labels_


def write_clusters_to_file(filename, clusters):
    f = open(filename, 'w')
    f.write('sha256,cluster\n')
    i = 0
    filename = get_filename(True)
    #print filename
    with open(filename) as data_file:
        for row in data_file:
            j = json.loads(row)
            if j["label"]=="malicious":
                f.write('%s' % j["sha256"])
                f.write(',')
                f.write('%s' % clusters[i])
                f.write('\n')
                i = i + 1
        for testrow in Test_Malicious_list:
                f.write('%s' % testrow)
                f.write(',')
                f.write('%s' % clusters[i])
                f.write('\n')
                i = i + 1
    print i
    f.close()


def write_clusters_to_file(filename, clusters, sha256):
    f = open(filename, 'w')
    f.write('sha256,cluster\n')
    #print filename
    for i in range(0, np.shape(sha256)[0]):
        f.write('%s' % sha256[i])
        f.write(',')
        f.write('%s' % clusters[i])
        f.write('\n')
    f.close()

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def set_test_data():
    with open('data/RandomForest_output.csv', 'rb') as csvfile:
        print csvfile
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        iterator = 0
        for row in spamreader:
            if row[1] == 'benign':
                Test_Benign_list.append(iterator)
            if row[1] == 'malicious':
                Test_Malicious_list.append(row[0])
            iterator = iterator + 1
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
    global Test_Benign_list
    global Test_Malicious_list
    global X_total

    reset_extended = False
    undersample = False
    remove_benign = True
    Test_Benign_list = []
    Test_Malicious_list = []

    # set_test_data()
    # print len(Test_Malicious_list)
    # print len(Test_Benign_list)
    # reset_data(reset_extended=reset_extended,with_undersampling=undersample,remove_benign=remove_benign)
    # X_total = np.append(X_train,X_test,axis=0)

    objdumphandler = ObjDumpHandler("data/malicious_objdump_40000")
    sha256, X_total  = objdumphandler.parse_file()

    print X_total.shape



    # clustering = {#"Kmeans": KMeans(n_clusters=2, random_state=0, max_iter=3000),any??
    #               "DBSScan": DBSCAN()#, 4
    #               #"AffinityPropagation": AffinityPropagation() 1900
    #     }
    # for name, cluster in clustering.items():
    #         Clusters_Train =  cluster_data(True, cluster)
    #         print Clusters_Train.shape
    #         pca = PCA(n_components = 3)
    #         X_total = pca.fit_transform(X_total)
    #         fig = plt.figure(1, figsize=(8, 6))
    #         ax = Axes3D(fig, elev=-150, azim=110)
    #         ax.scatter(X_total[:, 0], X_total[:, 1], X_total[:, 2], s=30,c= Clusters_Train)
    #         ax.set_title("First three PCA directions - TOTAL")
    #         ax.set_xlabel("1st eigenvector")
    #         ax.w_xaxis.set_ticklabels([])
    #         ax.set_ylabel("2nd eigenvector")
    #         ax.w_yaxis.set_ticklabels([])
    #         ax.set_zlabel("3rd eigenvector")
    #         ax.w_zaxis.set_ticklabels([])
    #         plt.show()
    #         write_clusters_to_file(name+'_Objdump_Train.csv', Clusters_Train, sha256)
    #         print "writing Train"
    #         write_clusters_to_file(name+'_Train.csv', Clusters_Train)
    #         print "Train Done"

    Z = linkage(X_total, 'ward')
    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')

    fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,  # useful in small plots so annotations don't overlap
    )
    plt.show()
    #max_d = 100000 @ Objdump
    #clusters = fcluster(Z, max_d, criterion='distance')
    #clusters


if __name__ == "__main__":
    main()


            #"MLP500,250,10,100" : MLPClassifier(hidden_layer_sizes=(500,250,10,100)),
            #"MLP500,10,500" : MLPClassifier(hidden_layer_sizes=(500,10,500),warm_start=True, beta_1=0.999, beta_2=0.9900),
            #"MLP":MLPClassifier(batch_size=20, warm_start=True,beta_1=0.09, beta_2=0.09)
            #"DBSCAN":DBSCAN()
           # "MLPlogistic":MLPClassifier(hidden_layer_sizes=(500,100,10,2,500),activation='logistic',max_iter=1000),
            # "MLPtanh":MLPClassifier(activation='tanh',max_iter=1000),
            # "MLPidentity":MLPClassifier(activation='identity',max_iter=1000),
            # Baseline Traning: Accuracy of MLP is 91.71597 %
            # Accuracy of MLPlogistic is 87.8822 %
            # Accuracy of MLPtanh is 90.93701 %
            # Accuracy of MLP is 91.6445 %
            # Accuracy of MLPidentity is 86.42633 %
            # "KNN Classifier": KNeighborsClassifier(n_neighbors=5, metric='euclidean', p = 2),
            # "Logistic Regression": LogisticRegression(),
            # "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
            #"SVCsigmoid" : SVC(kernel='sigmoid',C=5.0),
            #"SVCpoly":SVC(kernel='poly',degree= 2,C=3.0),
            #"SVC" : SVC(C=3.0)
