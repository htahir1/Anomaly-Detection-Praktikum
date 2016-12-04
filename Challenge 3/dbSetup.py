# -*- encoding: utf-8 -*-

import sys, sqlite3, json, os
from dbInterface import dbInterface, dbTools
import numpy as np
import pandas as pd

data_dir = 'data/'
data_name = 'network_data'
data_features = data_dir + 'features.csv'
data_testing = data_dir + 'testing-set.csv'
data_training = data_dir + 'training-set.csv'

data_db = data_dir + data_name + '.db'

dbi = None  # initialized in run
dbiTools = None  # initialized in run

table_training = 'training'
table_testing = 'testing'
table_features = 'features'

datatype_nominal = 'TEXT'
datatype_integer = 'INTEGER'
datatype_float = 'REAL'
datatype_binary = 'NUMERIC'
datatype_timestamp = 'NUMERIC'


table_ids = {table_training, table_testing, table_features}


def closeSQLConnection():
    dbi.close()


def dataTypeReplacement(csv_val):
    return{
        'nominal': datatype_nominal,
        'integer': datatype_integer,
        'Integer': datatype_integer,
        'Float': datatype_float,
        'Binary': datatype_binary,
        'binary': datatype_binary,
        'Timestamp': datatype_timestamp
    }[csv_val]

def getTableColumns(feature_data):
    table_columns = []
    table_column_data_type = []

    for x in feature_data['Name']:
        table_columns.append(x)

    for x in feature_data['Type ']:
        table_column_data_type.append(dataTypeReplacement(x))

    query = ''
    i = 0
    for x in table_columns:
        query += x + ' ' + table_column_data_type[i] + ', '
        i += 1
    query = (query[:-2]+' ')
    return query


def setupTables(dbcursor, table_columns):

    # Table 1
    queryA = ('CREATE TABLE IF NOT EXISTS ' + table_training + '('
               ''+table_columns
              + ');')

     # Table 2
    queryB = ('CREATE TABLE IF NOT EXISTS ' + table_testing + '('
               ''+table_columns
              + ');')

    # execute all queries
    for query in [ queryA, queryB]:
        dbcursor.execute(query)


def import_data():
    testing_data = pd.read_csv(data_testing, sep=',', error_bad_lines=False, encoding='utf-8')
#
    training_data = pd.read_csv(data_training, sep=',', error_bad_lines=False, encoding='utf-8')

    features_data = pd.read_csv(data_features, sep=',', error_bad_lines=False)


    print list(testing_data.columns.values)
    print list(training_data.columns.values)
    print list(features_data.columns.values)


    return training_data, testing_data, features_data


def initSQLConnection():
    global dbi, dbitools

    # SQL init stuff
    dbi = dbInterface(data_db)
    dbitools = dbTools(dbi.remoteCommit, {}, table_ids)


def setupDatabase():
    global dbi, dbitools

    # deleting existing db to retain data consistency
    if os.path.isfile(data_db):
        os.remove(data_db)

    training_data, testing_data, feature_data = import_data()

    table_columns = getTableColumns(feature_data)
    initSQLConnection()

    setupTables(dbi.getCursor(),table_columns)
    training_data.to_sql(table_training, dbi._conn, flavor='sqlite', if_exists='replace', index=False, chunksize=5)
    testing_data.to_sql(table_testing, dbi._conn, flavor='sqlite', if_exists='replace', index=False, chunksize=5)
  #  feature_data.to_sql(table_features, dbi._conn, flavor='sqlite', if_exists='replace', index=False, chunksize=5)

    dbi.close()


def getFeatures(train_mode):
    if train_mode:
        reviewer_data = np.delete(np.array([list(elem)[1:] for elem in (getFeatureReviewers(train_mode)[0])], dtype=np.float), -1, 1)
        review_data = np.array([list(elem)[1:] for elem in (getFeatureReviewsCounts(train_mode)[0])], dtype=np.float)
        # reviewer_length_data = np.delete(np.array([list(elem)[1:] for elem in (getFeatureReviewLength(train_mode)[0])], dtype=np.float), -1, 1)
        # reviewer_percent_positive_reviews_data = np.delete(np.array([list(elem)[1:] for elem in (getFeaturePercentPositiveReviews(train_mode)[0])], dtype=np.float), -1, 1)
        # reviewer_average_rating_data = np.delete(np.array([list(elem)[2:] for elem in (getFeatureAvgRatingByHotel(train_mode)[0])], dtype=np.float), -1, 1)
        # reviewer_reviews_day_data = np.array([list(elem)[1:] for elem in (getFeatureReviewsPerDay(train_mode)[0])], dtype=np.float)
    else:
        reviewer_data = np.array([list(elem)[1:] for elem in (getFeatureReviewers(train_mode)[0])], dtype=np.float)
        review_data = np.array([list(elem)[1:] for elem in (getFeatureReviewsCounts(train_mode)[0])], dtype=np.float)

        # reviewer_length_data = np.array([list(elem)[1:] for elem in (getFeatureReviewLength(train_mode)[0])], dtype=np.float)
        # reviewer_percent_positive_reviews_data = np.array([list(elem)[1:] for elem in (getFeaturePercentPositiveReviews(train_mode)[0])], dtype=np.float)
        # reviewer_average_rating_data = np.array([list(elem)[2:] for elem in (getFeatureAvgRatingByHotel(train_mode)[0])], dtype=np.float)
        # reviewer_reviews_day_data = np.array([list(elem)[1:] for elem in (getFeatureReviewsPerDay(train_mode)[0])], dtype=np.float)

    data = np.concatenate((reviewer_data, review_data), axis=1)

    return data
    
class BigTransaction():
    # how to:
    # 
    # migration = subClassOfBigTransactions()
    # for ...
    #   migration.<migrate>(...)
    # migration.finish()
    
    def __init__(self, commitAfter=-1):
        self.counter = 0
        self.commitAfter = commitAfter
        dbitools.suspendCommitting()    # turn off auto commit
        
    def finish(self):
        dbitools.unsuspendCommitting()    # turn on auto commit
        dbitools.commit()                 # final commit
    
    def trackMigration(self):
        self.counter = self.counter + 1
        
        if self.commitAfter > 0:
            if self.counter % self.commitAfter == 0:
                dbitools.commit()


if __name__ == "__main__":
    # setupDatabase()
    pass
