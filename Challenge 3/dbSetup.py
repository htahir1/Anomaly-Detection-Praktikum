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


table_hotel = 'hotel'
table_hotel_hotelID = 'hotelID'
table_hotel_location = 'location'
table_hotel_reviewCount = 'reviewCount'
table_hotel_rating = 'rating'
table_hotel_categories = 'categories'
table_hotel_address = 'address'
table_hotel_AcceptsCreditCards = 'AcceptsCreditCards'
table_hotel_PriceRange = 'PriceRange'
table_hotel_WiFi = 'WiFi'
table_hotel_webSite = 'webSite'
table_hotel_phoneNumber = 'phoneNumber'
table_hotel_filReviewCount = 'filReviewCount'

table_reviewer = 'reviewer'
table_reviewer_reviewerID = 'reviewerID'
table_reviewer_name = 'name'
table_reviewer_location = 'location'
table_reviewer_yelpJoinDate = 'yelpJoinDate'
table_reviewer_friendCount = 'friendCount'
table_reviewer_reviewCount = 'reviewCount'
table_reviewer_firstCount = 'firstCount'
table_reviewer_usefulCount = 'usefulCount'
table_reviewer_coolCount = 'coolCount'
table_reviewer_funnyCount = 'funnyCount'
table_reviewer_complimentCount = 'complimentCount'
table_reviewer_tipCount = 'tipCount'
table_reviewer_fanCount = 'fanCount'

table_reviews_test = 'reviews_test'
table_reviews_test_id = 'id'
table_reviews_test_date = 'date'
table_reviews_test_reviewID = 'reviewID'
table_reviews_test_reviewerID = 'reviewerID'
table_reviews_test_reviewContent = 'reviewContent'
table_reviews_test_rating = 'rating'
table_reviews_test_usefulCount = 'usefulCount'
table_reviews_test_coolCount = 'coolCount'
table_reviews_test_funnyCount = 'funnyCount'
table_reviews_test_hotelID = 'hotelID'


table_reviews_train = 'reviews_train'
table_reviews_train_date = 'date'
table_reviews_train_reviewID = 'reviewID'
table_reviews_train_reviewerID = 'reviewerID'
table_reviews_train_reviewContent = 'reviewContent'
table_reviews_train_rating = 'rating'
table_reviews_train_usefulCount = 'usefulCount'
table_reviews_train_coolCount = 'coolCount'
table_reviews_train_funnyCount = 'funnyCount'
table_reviews_train_fake = 'fake'
table_reviews_train_hotelID = 'hotelID'

table_ids = {table_training, table_testing, table_features}


def closeSQLConnection():
    dbi.close()


def setupTables(dbcursor):

    # Table 1
    queryA = ('CREATE TABLE IF NOT EXISTS ' + table_hotel + '('
              '' + table_hotel_hotelID + ' TEXT PRIMARY KEY, '
              '' + table_hotel_location + ' TEXT, '
              '' + table_hotel_reviewCount + ' REAL, '
              '' + table_hotel_rating + ' TEXT, '
              '' + table_hotel_categories + ' TEXT, '
              '' + table_hotel_address + ' TEXT, '
              '' + table_hotel_AcceptsCreditCards + ' TEXT, '
              '' + table_hotel_PriceRange + ' TEXT, '
              '' + table_hotel_WiFi + ' TEXT, '
              '' + table_hotel_webSite + ' TEXT, '
              '' + table_hotel_phoneNumber + ' TEXT, '
              '' + table_hotel_filReviewCount + ' INTEGER '
              + ');')

    # Table 2
    queryB = ('CREATE TABLE IF NOT EXISTS ' + table_reviewer + '('
              '' + table_reviewer_reviewerID + ' TEXT PRIMARY KEY, '
              '' + table_reviewer_name + ' TEXT, '
              '' + table_reviewer_location + ' TEXT, '
              '' + table_reviewer_yelpJoinDate + ' TEXT, '
              '' + table_reviewer_friendCount + ' INTEGER, '
              '' + table_reviewer_reviewCount + ' INTEGER, '
              '' + table_reviewer_firstCount + ' INTEGER, '
              '' + table_reviewer_usefulCount + ' INTEGER, '
              '' + table_reviewer_coolCount + ' INTEGER, '
              '' + table_reviewer_funnyCount + ' INTEGER, '
              '' + table_reviewer_complimentCount + ' INTEGER, '
              '' + table_reviewer_tipCount + ' INTEGER, '
              '' + table_reviewer_fanCount + ' INTEGER '
              + ');')

    # Table 3
    queryC = ('CREATE TABLE IF NOT EXISTS ' + table_reviews_test + '('
              '' + table_reviews_test_id + ' INTEGER, '
              '' + table_reviews_test_date + ' TEXT, '
              '' + table_reviews_test_reviewID + ' TEXT PRIMARY KEY, '
              '' + table_reviews_test_reviewerID + ' TEXT, '
              '' + table_reviews_test_reviewContent + ' TEXT, '
              '' + table_reviews_test_rating + ' REAL, '
              '' + table_reviews_test_usefulCount + ' INTEGER, '
              '' + table_reviews_test_coolCount + ' INTEGER, '
              '' + table_reviews_test_funnyCount + ' INTEGER, '
              '' + table_reviews_test_hotelID + ' TEXT, '
              + 'FOREIGN KEY(' + table_reviews_test_reviewerID + ') REFERENCES ' + table_reviewer + '(' + table_reviewer_reviewerID + '), '
              + 'FOREIGN KEY(' + table_reviews_test_hotelID + ') REFERENCES ' + table_hotel + '(' + table_hotel_hotelID + ') '
              + ');')

    # Table 4
    queryD = ('CREATE TABLE IF NOT EXISTS ' + table_reviews_train + '('
             '' + table_reviews_train_date + ' TEXT, '
             '' + table_reviews_train_reviewID + ' TEXT PRIMARY KEY, '
             '' + table_reviews_train_reviewerID + ' TEXT, '
             '' + table_reviews_train_reviewContent + ' TEXT, '
             '' + table_reviews_train_rating + ' REAL, '
             '' + table_reviews_train_usefulCount + ' INTEGER, '
             '' + table_reviews_train_coolCount + ' INTEGER, '
             '' + table_reviews_train_funnyCount + ' INTEGER, '
             '' + table_reviews_train_fake + ' INTEGER, '
             '' + table_reviews_train_hotelID + ' TEXT, '
             + 'FOREIGN KEY(' + table_reviews_train_reviewerID + ') REFERENCES ' + table_reviewer + '(' + table_reviewer_reviewerID + '), '
             + 'FOREIGN KEY(' + table_reviews_train_hotelID + ') REFERENCES ' + table_hotel + '(' + table_hotel_hotelID + ') '
             + ');')


    
    # execute all queries
    for query in [queryA, queryB, queryC, queryD]:
        dbcursor.execute(query)


def import_data():
    testing_data = pd.read_csv(data_testing, sep=',', error_bad_lines=False, encoding='utf-8')

    training_data = pd.read_csv(data_training, sep=',', error_bad_lines=False, encoding='utf-8')

    features_data = pd.read_csv(data_features, sep=',', error_bad_lines=False, encoding='utf-8')


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

    initSQLConnection()

    # setupTables(dbi.getCursor())

    training_data, testing_data, feature_data = import_data()

    training_data.to_sql(table_training, dbi._conn, flavor='sqlite', if_exists='replace', index=False, chunksize=5)
    testing_data.to_sql(table_testing, dbi._conn, flavor='sqlite', if_exists='replace', index=False, chunksize=5)
    feature_data.to_sql(table_features, dbi._conn, flavor='sqlite', if_exists='replace', index=False, chunksize=5)

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
