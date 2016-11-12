import sqlite3, math


class dbInterface(object):
    
    def __init__(self, pathToFile):
        #checking if pathToFile avaiable
        assert pathToFile

        #setting up connection, important to set row_factory before getting a cursor instance
        self._conn = sqlite3.connect(pathToFile)
        # self._conn.text_factory = str
        # self._conn.row_factory = sqlite3.Row        #content of returned mapping is accessible via key or index
        
        self._curs = self._conn.cursor()
        
    
    def close(self):
        self._conn.close()
        
        
    def getCursor(self):
        return self._curs
        
        
    def remoteCommit(self):
        self._conn.commit()


class dbTools(object):
    
    def __init__(self, commitCallback, convertTable, idTable):
        assert commitCallback
        #assert convertTable
        assert idTable
        
        self.suspendCommit = False
        self.commitCallback = commitCallback
        self.convertBack = convertTable
        self.idColumn = idTable
    
    def commit(self):
        if not self.suspendCommit:
            self.commitCallback()
    
    def suspendCommitting(self):
        self.suspendCommit = True
        print '## disabling auto commit - big transaction upcoming'
        
    def unsuspendCommitting(self):
        self.suspendCommit = False
        print '## enabling auto commit - big transaction done'
    

    def update(self, cursor, table, whereQuery, data):
        #data is a dictionary: {"columnName": "value", ...}
                
        query = "UPDATE " + table + " SET " + ','.join(self.toSQLUpdateParameters(data)) + ' ' + whereQuery + ';';
        
        cursor.execute(query)
        self.commit()
    
    
    def updateRow(self, cursor, table, identifier, data):
        columnWithIds = self.getIdColumn(table)
        
        if columnWithIds is not None:
            return self.update(cursor, table,  "WHERE " + columnWithIds + '=' + str(identifier), data)
        
        else:
            return None
    
    
    def insert(self, cursor, table, data):
        #data is a dictionary: {"columnName": "value", ...}
        
        params = self.toSQLInsertParamters(data)
        query = "INSERT INTO " + table + "(" + params["keys"] + ") VALUES(" + params["values"] + ");"
        
        cursor.execute(query)
        self.commit()
        
        return cursor.lastrowid
    
    
    def delete(self, cursor, table, whereQuery):
        query = "DELETE FROM " + table + ' ' + whereQuery + ';';
        
        cursor.execute(query)
        self.commit()
    
    
    def deleteRow(self, cursor, table, identifier):
        columnWithIds = self.getIdColumn(table)
        
        if columnWithIds is not None:
            self.delete(cursor, table, "WHERE " + columnWithIds + '=' + str(identifier))
        
        else:
            return None
    
    
    def retrieve(self, cursor, table, whereQuery="", columns=[], orderby="", limitOffset=""):
        columnSelector = self.getColumnSelector(columns)
        
        query = "SELECT " + columnSelector + " FROM " + table + ' ' + whereQuery + ' ' + orderby + ' ' + limitOffset + ';'
        cursor.execute(query)
        
        return(self.toOriginalDataType(cursor.fetchall()))

    def retrieveResultsWithNames(self, cursor, table):
        query = ( "SELECT " + table + ".personId1, " + table + ".personId2, " + table +
                  ".similarity, people1.mail as mail1, people1.first_name as first_name1, people1.last_name as last_name1, "
                  "people2.mail as mail2, people2.first_name as first_name2, people2.last_name as last_name2 FROM "
                  "(people AS people1 JOIN " + table + " ON people1.id == " + table +
                  ".personId1) JOIN people AS people2 ON people2.id == " + table + ".personId2 WHERE people1.chair != people2.chair;" )

        cursor.execute(query)

        return(self.toOriginalDataType(cursor.fetchall()))

    def retrieveRow(self, cursor, table, identifier, columns=[], orderby=""):
        columnWithIds = self.getIdColumn(table)
        
        if columnWithIds is not None:
            columns = self.retrieve(cursor, table, "WHERE " + columnWithIds + '=' + str(identifier), columns, orderby, "LIMIT 1")
            
            if len(columns) != 0:
                return columns[0]
            else:
                return None
        
        else:
            return None
    
    
    def count(self, cursor, table, column="", whereQuery=""):
        #counting is only performable with a maximum of one column        
        if not column:
            column = '*'
        else:
            if column.find(',') is not -1:
                raise Exception("only one column allowed")
        
        query = "SELECT COUNT(" + column + ") FROM " + table
        
        if whereQuery:
            query += ' ' + whereQuery + ';'
        else:
            query += ';'
                
        cursor.execute(query)
        
        return cursor.fetchone()[0]
    
    
    def toOriginalDataType(self, data):
        #expects array of dicts of a SQL-query as input
        
        if data is None or len(data) is 0:
            return data
        
        else:
            #formerly in each loop: conKeys = self.containsKeys(row)
            conKeys = self.containsKeys(data[0])
            conData = []
            tempRow = None
            
            for row in data:
                #setting up temporary dictionary
                tempRow = {}
                
                #filling temporary dictionary with converted values and / or original output values
                for key in row.keys():
                    if key in conKeys:
                        tempRow[key] = self.convertBack[key](row[key])
                    else:
                        tempRow[key] = row[key]
                
                #appending filled temporary dictionary to converted data array
                conData.append(tempRow)            
            
            return conData
        
        
    def containsKeys(self, row):
        convert = []
        
        for key in row.keys():
            if key in self.convertBack.keys():
                convert.append(key)
        
        return convert
             
         
    def toSQLUpdateParameters(self, data):
        #Update Syntax: UPDATE <table> SET key=value, ... (WHERE ...)
        #returns array of n strings: ["key=value", ...]
        keys = data.keys()
        size = len(keys)
        
        parameters = [None] * size
        
        for i in range(0, size):
            parameters[i] = keys[i] + '=' + self.toSQLDataType(data[keys[i]])
        
        return parameters
    
    
    def toSQLInsertParamters(self, data):
        #Insert Syntax: INSERT INTO <table>(key, key, ...) VALUES(value, value, ...)
        #returns array of 2 strings: ["key,key,key,..", "value,value,value,..."]
        keys = data.keys()
        size = len(keys)
        
        parameters = {"keys": "", "values": ""}
        
        if size > 1:
            for i in range(0, size - 1):
                parameters["keys"] = parameters["keys"] + str(keys[i]) + ','
                parameters["values"] = parameters["values"] + self.toSQLDataType(data[keys[i]]) + ','
            
            #last vals without ','
            parameters["keys"] = parameters["keys"] + str(keys[i + 1])
            parameters["values"] = parameters["values"] + self.toSQLDataType(data[keys[i + 1]])
            
        else:
            parameters["keys"] = str(keys[0])
            parameters["values"] = parameters["values"] + self.toSQLDataType(data[keys[0]])
        
        return parameters
    
    
    def toSQLDataType(self, variable):
        t = type(variable)
        
        if t in [int, long, float]:
            return str(variable)
        
        elif t is bool:
            if variable:
                return "1"
            else:
                return "0"
            
        elif t is str:
            return "'" + self.SQLespace(variable) + "'"
        
        elif t is unicode:
            return "'" + self.SQLespace(variable) + "'"
			#return "'".encode("utf-8") + self.SQLespace(variable).encode("utf-8") + "'".encode("utf-8")
        
        else:
            return "NULL"
    
    
    def SQLespace(self, string):
        return string.replace("'", "''")
    
    
    def getColumnSelector(self, columns):
        #expects array of columns
        columnSelector = '*'
        
        #change if array is not empty
        if columns:
            columnSelector = ','.join(columns)
        
        return columnSelector
    
    
    def getIdColumn(self, table):
        try:
            return self.idColumn[table]
        except KeyError, e:
            print e
            #logging?
            return None
    
    
    def idExists(self, cursor, table, identifier):
        columnWithIds = self.getIdColumn(table)
        
        if columnWithIds is not None:
            cursor.execute("SELECT EXISTS(SELECT 1 FROM ints WHERE " + columnWithIds + '=' + str(identifier) + " LIMIT 1);")
            #db returns "(0,)" or "(1,)" from the EXIST request
            return (cursor.fetchone()[0] == True)
        
        else:
            return False
    
    
    def idOfLastEntry(self, cursor, table):
        #SELECT MAX(id) AS member_id, name, value FROM YOUR_TABLE_NAME
        
        columnWithIds = self.getIdColumn(table)
        
        if columnWithIds is not  None:
            cursor = "SELECT MAX(" + columnWithIds + ") AS id FROM " + table
            return cursor.fetchone()[0]["id"]
        
        else:
            return -1

