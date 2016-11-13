# -*- encoding: utf-8 -*-

import sys, sqlite3, json, os
from dbInterface import dbInterface, dbTools
import numpy
import pandas as pd

data_dir = 'challenge_data/'
data_name = 'yelp_data'
data_hotel = data_dir + 'yelp_data_hotel.dat'
data_reviewer = data_dir + 'yelp_data_reviewer.dat'
data_review_test = data_dir + 'yelp_data_test.dat'
data_review_train = data_dir + 'yelp_data_train.dat'

data_db = data_dir + data_name + '.db'

dbi = None  # initialized in run
dbiTools = None  # initialized in run

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

table_ids = {table_hotel, table_reviewer, table_reviews_train, table_reviews_test}


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
    testing_data = pd.read_csv(data_review_test, sep=';', error_bad_lines=False, encoding='utf-8')

    training_data = pd.read_csv(data_review_train, sep=';', error_bad_lines=False, encoding='utf-8')

    reviewer_data = pd.read_csv(data_reviewer, sep=';', error_bad_lines=False, encoding='utf-8')

    hotel_data = pd.read_csv(data_hotel, sep=';', error_bad_lines=False, encoding='utf-8')

    print list(testing_data.columns.values)
    print list(training_data.columns.values)
    print list(reviewer_data.columns.values)
    print list(hotel_data.columns.values)

    return training_data, testing_data, reviewer_data, hotel_data


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

    initSQLConnection();

    setupTables(dbi.getCursor())

    training_data, testing_data, reviewer_data, hotel_data = import_data()

    reviewer_data.to_sql(table_reviewer, dbi._conn, flavor='sqlite', if_exists='replace', index=False, chunksize=5)
    hotel_data.to_sql(table_hotel, dbi._conn, flavor='sqlite', if_exists='replace', index=False, chunksize=5)
    training_data.to_sql(table_reviews_train, dbi._conn, flavor='sqlite', if_exists='replace', index=False, chunksize=5)
    testing_data.to_sql(table_reviews_test, dbi._conn, flavor='sqlite', if_exists='replace', index=False, chunksize=5)

    dbi.close()


def getAllPersonIds():
    entries = dbitools.retrieve(dbi.getCursor(), table_people, '', [table_people_id])
    return [entry[table_people_id] for entry in entries]


def getPerson(id):
    return dbitools.retrieveRow(dbi.getCursor(), table_people, id)

def addOccupation(title):
    return dbitools.insert(dbi.getCursor(), table_occupation, {table_occupation_title: title})


def addPerson(data):
    return dbitools.insert(dbi.getCursor(), table_people, data)

def getOccupationId(title):
    row = dbitools.retrieve(dbi.getCursor(), table_occupation,
                            'WHERE ' + table_occupation_title + '=' + "'" + title + "'")

    if len(row) == 0:
        return None
    else:
        return row[0][table_occupation_id]


def showProgress(current, max, msg):
    sys.stdout.write('' + str(current) + '/' + str(max) +  ' ' + msg + '\r')
    # print ('' + str(current) + '/' + str(max) +  ' ' + msg + '\r')
    sys.stdout.flush()

    
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
