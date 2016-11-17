# -*- encoding: utf-8 -*-

import sys, sqlite3, json, os
from dbInterface import dbInterface, dbTools
import numpy as np
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
    print "Importing test data from csv to database.."
    testing_data = pd.read_csv(data_review_test, sep=';', error_bad_lines=False, encoding='utf-8')

    print "Importing train data from csv to database.."
    training_data = pd.read_csv(data_review_train, sep=';', error_bad_lines=False, encoding='utf-8')

    print "Importing reviewer data from csv to database.."
    reviewer_data = pd.read_csv(data_reviewer, sep=';', error_bad_lines=False, encoding='utf-8')

    print "Importing hotel data from csv to database.."
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


def getHotelIds():
    query = "SELECT hotel_id FROM " + table_hotel
    dbi.getCursor().execute(query)
    return dbi.getCursor().fetchall()


def getFeaturesByReview(train_mode):
    if train_mode:
        table = table_reviews_train
    else:
        table = table_reviews_test
    query = "SELECT rt5.funnyCount, rt5.reviewCount, rt5.firstCount, rt5.usefulCount, rt5.coolCount, rt5.complimentCount, rt5.fanCount, rt5.tipCount, (LENGTH(rt.reviewContent)- LENGTH(REPLACE(rt.reviewContent, ' ', ''))) reviewLength, rt2.ratingAbv3 as ratingAbv3, rt2.allCount as allRatingsCount,"
    query = query + " rt.rating, rt3.avgRatingByHotel as avgRatingByHotel, rt4.maximumNumReviewsPerDay as maximumNumReviewsPerDay "
    if train_mode:
        query = query + " ,rt.fake as fake "
    query = query + "FROM "+ table+" as rt "
    query = query + " LEFT JOIN ( SELECT reviewerID, COUNT(CASE WHEN rating >3 THEN 1 ELSE NULL END) as ratingAbv3, COUNT(*) as allCount FROM " + table + " GROUP BY reviewerID) as rt2 ON rt.reviewerID = rt2.reviewerID "
    query = query + " LEFT JOIN ( SELECT reviewerID,hotelID, AVG(rating) as avgRatingByHotel FROM "+table+" GROUP BY hotelID) as rt3 ON rt.hotelID = rt3.hotelID "
    query = query + " LEFT JOIN ( SELECT rt1.reviewID, rt1.date, rt4.maximumNumReviewsPerDay FROM "+table+" rt1 INNER JOIN (SELECT rev_t.reviewID, rev_t.reviewerID, date, COUNT (DISTINCT reviewID) as 'maximumNumReviewsPerDay' FROM "
    query = query +table +" rev_t GROUP BY rev_t.reviewerID, date) as rt4 ON rt1.date = rt4.date AND rt1.reviewerID = rt4.reviewerID ) as rt4 ON rt.reviewID = rt4.reviewID LEFT JOIN reviewer as rt5 ON rt5.reviewerID = rt.reviewerID"
    print query
    dbi.getCursor().execute(query)
    headers = list()
    headers.append(table_reviews_test_reviewID)
    headers.append('reviewLength')
    headers.append('percentPositiveReviews') #above 3
    headers.append('reviewerDeviation')
    headers.append('maxNumberofReviews')
    if train_mode:
        headers.append('label')
    ret = dbi.getCursor().fetchall()

    processed = []
    # for item in ret:
    #     tmp_list = list(item)
    #     if None not in tmp_list:
    #         tmp_list[2] = float(tmp_list[2])/tmp_list[3]
    #         tmp_list.pop(3)
    #         item = tuple(tmp_list)
    #         tmp_list[3] = tmp_list[3] - tmp_list[4]#rating - avg
    #         tmp_list.pop(4)
    #         processed.append(tmp_list)
    return (ret, headers)


def getFeatureReviewers(train_mode):
    if train_mode:
        table = table_reviews_train
    else:
        table = table_reviews_test

    query = "SELECT IFNULL(a.reviewID,'Outlier') as reviewID, b.funnyCount funnyCount, b.reviewCount reviewCount, b.firstCount firstCount, b.usefulCount usefulCount, b.coolCount coolCount, b.complimentCount complimentCount, b.fanCount fanCount, b.tipCount tipCount"
    if train_mode:
        query = query + " ,fake "
    query = query + " FROM ((SELECT reviewID, reviewerID "

    if train_mode:
        query = query + " ,(CASE WHEN fake = 'N' THEN 0 WHEN fake = 'Y' THEN 1 ELSE fake END) as fake "

    query = query + " FROM "+table+" ) a LEFT JOIN (SELECT rt.reviewerID,rt.funnyCount, rt.reviewCount, rt.firstCount, rt.usefulCount, rt.coolCount, rt.complimentCount, rt.fanCount, rt.tipCount FROM reviewer rt WHERE reviewerID IN "
    query = query + "(SELECT reviewerID FROM "+table+")) b ON a.reviewerID = b.reviewerID ) ;"

    headers = list()
    dbi.getCursor().execute(query)
    columns = dbi.getCursor().description
    for column in columns:
        headers.append(column[0])

    ret = dbi.getCursor().fetchall()
    return (ret, headers)


def getFeatureReviewLength(train_mode):
    if train_mode:
        table = table_reviews_train
    else:
        table = table_reviews_test

    query = "SELECT IFNULL(reviewID,'Outlier') as reviewID, (LENGTH(rt.reviewContent)- LENGTH(REPLACE(reviewContent, ' ', ''))) reviewLength"



    if train_mode:
        query = query + " ,(CASE WHEN fake = 'N' THEN 0 WHEN fake = 'Y' THEN 1 ELSE fake END) as fake "

    query = query + " FROM "+table+" as rt ;"

    headers = list()
    dbi.getCursor().execute(query)
    columns = dbi.getCursor().description
    for column in columns:
        headers.append(column[0])

    ret = dbi.getCursor().fetchall()
    return (ret, headers)


def getFeaturePercentPositiveReviews(train_mode):
    if train_mode:
        table = table_reviews_train
    else:
        table = table_reviews_test

    query = "SELECT IFNULL(reviewID,'Outlier') as reviewID ,(b.ratingAbv3/(b.allCount + 0.0)) as PercentPositiveReviews, b.allcount as TotalReviewsByThisReviewer "
    if train_mode:
        query = query + " ,(CASE WHEN fake = 'N' THEN 0 WHEN fake = 'Y' THEN 1 ELSE fake END) as fake "

    query = query + " FROM ((SELECT reviewID, reviewerID"

    if train_mode:
        query = query + " ,(CASE WHEN fake = 'N' THEN 0 WHEN fake = 'Y' THEN 1 ELSE fake END) as fake "

    query = query + " FROM "+table+ ") a LEFT JOIN (SELECT reviewerID, COUNT(CASE WHEN rating >3 THEN 1 ELSE NULL END) as ratingAbv3, COUNT(*) as allCount FROM "+table+" GROUP BY reviewerID) b ON a.reviewerID = b.reviewerID) ;"

    headers = list()
    dbi.getCursor().execute(query)
    columns = dbi.getCursor().description
    for column in columns:
        headers.append(column[0])

    ret = dbi.getCursor().fetchall()
    return (ret, headers)


def getFeatureAvgRatingByHotel(train_mode):
    if train_mode:
        table = table_reviews_train
    else:
        table = table_reviews_test

    query = "SELECT IFNULL(reviewID,'Outlier') as reviewID ,b.* "
    if train_mode:
        query = query + " ,(CASE WHEN fake = 'N' THEN 0 WHEN fake = 'Y' THEN 1 ELSE fake END) as fake "

    query = query + " FROM ((SELECT reviewID, hotelID "

    if train_mode:
        query = query + " ,(CASE WHEN fake = 'N' THEN 0 WHEN fake = 'Y' THEN 1 ELSE fake END) as fake "

    query = query + " FROM "+table+") a LEFT JOIN (SELECT hotelID, AVG(rating) as avgRatingByHotel FROM "+table +" GROUP BY hotelID) b ON a.hotelID = b.hotelID) ;"

    headers = list()
    dbi.getCursor().execute(query)
    columns = dbi.getCursor().description
    for column in columns:
        headers.append(column[0])

    ret = dbi.getCursor().fetchall()
    return (ret, headers)


def getFeatureReviewsPerDay(train_mode):
    if train_mode:
        table = table_reviews_train
    else:
        table = table_reviews_test

    query = "SELECT IFNULL(a.reviewID,'Outlier') as reviewID ,b.maximumNumReviewsPerDay as MaximumNumReviewsPerDay ";
    if train_mode:
        query = query + " ,(CASE WHEN fake = 'N' THEN 0 WHEN fake = 'Y' THEN 1 ELSE fake END) as fake "

    query = query + " FROM ((SELECT * FROM "+table+") a LEFT JOIN (SELECT rev_t.reviewerID, date, COUNT (DISTINCT IFNULL(reviewID,1)) as 'maximumNumReviewsPerDay' FROM "+table+" rev_t GROUP BY rev_t.reviewerID, date) b ON a.reviewerID = b.reviewerID AND a.date =b.date) ;"

    headers = list()
    dbi.getCursor().execute(query)
    columns = dbi.getCursor().description
    for column in columns:
        headers.append(column[0])

    ret = dbi.getCursor().fetchall()
    return (ret, headers)


def concatenateFeatures(a1, a2):
    pass


def getFeatures(train_mode):
    if train_mode:
        reviewer_data = np.array([list(elem)[1:] for elem in (getFeatureReviewers(train_mode)[0])], dtype=np.float)
        # reviewer_length_data = np.delete(np.array([list(elem)[1:] for elem in (getFeatureReviewLength(train_mode)[0])], dtype=np.float), -1, 1)
        # reviewer_percent_positive_reviews_data = np.delete(np.array([list(elem)[1:] for elem in (getFeaturePercentPositiveReviews(train_mode)[0])], dtype=np.float), -1, 1)
        # reviewer_average_rating_data = np.delete(np.array([list(elem)[2:] for elem in (getFeatureAvgRatingByHotel(train_mode)[0])], dtype=np.float), -1, 1)
        # reviewer_reviews_day_data = np.array([list(elem)[1:] for elem in (getFeatureReviewsPerDay(train_mode)[0])], dtype=np.float)
    else:
        reviewer_data = np.array([list(elem)[1:] for elem in (getFeatureReviewers(train_mode)[0])], dtype=np.float)
        # reviewer_length_data = np.array([list(elem)[1:] for elem in (getFeatureReviewLength(train_mode)[0])], dtype=np.float)
        # reviewer_percent_positive_reviews_data = np.array([list(elem)[1:] for elem in (getFeaturePercentPositiveReviews(train_mode)[0])], dtype=np.float)
        # reviewer_average_rating_data = np.array([list(elem)[2:] for elem in (getFeatureAvgRatingByHotel(train_mode)[0])], dtype=np.float)
        # reviewer_reviews_day_data = np.array([list(elem)[1:] for elem in (getFeatureReviewsPerDay(train_mode)[0])], dtype=np.float)

    # data = np.concatenate((reviewer_data, reviewer_length_data, reviewer_percent_positive_reviews_data, reviewer_average_rating_data, reviewer_reviews_day_data), axis=1)

    return reviewer_data

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
