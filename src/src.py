'''
Learning From Data
Term Project
25.05.2018
150130048   Furkan Artunç
150130901   Ibrahim Aliu
'''

from pandas import read_csv, DataFrame
from numpy import column_stack, nan, argsort, array
from numpy.random import randint
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from warnings import filterwarnings
from sklearn import svm
filterwarnings("ignore")


class LfdCompetitor():
    def __init__(self, train_data_file_path, test_data_file_path):
        self.train_data_path, self.test_data_path = train_data_file_path, test_data_file_path
        self.test_X, self.train_X, self.train_y = None, None, None
        self.columns = ['orderItemID', 'orderDate', 'deliveryDate', 'itemID', 'size', 'color', 'manufacturerID',
                   'price', 'customerID', 'salutation', 'dateOfBirth', 'state', 'creationDate',
                   'returnShipment']
        self.encoders = []
        self.model = None

    '''
    Read both train and test data and cleans them, set it to related variables of class, encode categorical values
    '''
    def read_data_and_encode(self):
        columns = self.columns
        to_replace = ['?', '', '-', '.']
        train_data = read_csv(self.train_data_path, sep=',', usecols=columns).replace(to_replace, nan).dropna()
        train_data['size'] = train_data['size'].str.lower()
        train_data['color'] = train_data['color'].str.lower()
        test_data = read_csv(self.test_data_path, sep=',', usecols=columns[:13])
        self.fill_blanks(test_data, to_replace, dataset='test')
        test_data['size'] = test_data['size'].str.lower()
        test_data['color'] = test_data['color'].str.lower()

        train_data = self.clean_unsized_labels(train_data)
        test_data = self.clean_unsized_labels(test_data)

        encoder = LabelEncoder()
        self.encoders.append(encoder.fit(train_data[columns[0]]))
        self.encoders.append(encoder.fit(train_data[columns[1]]))

        X = column_stack((self.encoders[0].fit_transform(train_data[columns[0]].values), self.encoders[1].fit_transform(train_data[columns[1]].values)))
        for i, col in zip(range(2, 13), columns[2:13]):
            self.encoders.append(encoder.fit(train_data[col].values))
            X = column_stack((X, self.encoders[i].fit_transform(train_data[col].values)))
        self.train_y = train_data['returnShipment']
        self.train_X = DataFrame(X, columns=columns[:13])

        tst_x = column_stack((self.encoders[0].fit_transform(test_data[columns[0]].values), self.encoders[1].fit_transform(test_data[columns[1]].values)))
        for i, col in zip(range(2, 13), columns[2:13]):
            self.encoders.append(encoder.fit(test_data[col].values))
            tst_x = column_stack((tst_x, self.encoders[i].fit_transform(test_data[col].values)))
        self.test_X = DataFrame(tst_x, columns=columns[:13])

        print("Train & Test data is loaded.")

    '''Data clenaer for size feature'''
    def clean_unsized_labels(self, data):
        return data.replace('unsized', ['l', 'xl', 'm'][randint(0, 2)])

    '''
    Find most occured data into the set
    '''
    def mode(self, data):
        arr = dict.fromkeys(data, 0)
        for key in data:
            arr[key] += 1
        return list(arr.keys())[list(arr.values()).index(max(arr.values()))]
    '''
    Fill test data's dummy values with mode of the column
    '''
    def fill_blanks(self, data, to_replace, dataset):
        cols = self.columns
        if dataset == 'test':
            cols = self.columns[:13]
        for col in cols:
            values = list(data[col].values)
            val = self.mode(values)
            if val in to_replace:
                values = (filter((val).__ne__, values))
                val = self.mode(values)
            data[col] = data[col].replace(to_replace, val)

    '''
    Feature Selection, actually it reduces the accuracy
    '''
    def feature_selection(self):
        model = ExtraTreesClassifier()
        model.fit(self.train_X, self.train_y)
        cols = array(self.columns)[argsort(model.feature_importances_)[-8:][::-1]]
        return self.train_X[cols], self.test_X[cols]

    '''
    Train network with xgboost
    '''
    def train_xgboost(self, trainx):
        self.model = XGBClassifier(num_class=2, objective='multi:softmax')
        self.model.fit(trainx, self.train_y)

    '''
        Train network with random forest from sklearn
    '''
    def train_randforest(self, trainx):
        self.model = RandomForestClassifier()
        self.model.fit(trainx, self.train_y)

    '''
        Train network with logistic regression
    '''
    def train_logreg(self, trainx):
        self.model = LogisticRegression()
        self.model.fit(trainx, self.train_y)

    '''
        Train network with support vector machine
    '''
    def train_svm(self, trainx):
        self.model = svm.SVC()
        self.model.fit(trainx, self.train_y)

    '''
    Make predictions based on trained model
    '''
    def predict(self, testx):
        return self.model.predict(testx)

    '''
    Main functin for class, calls other functions and enables program flow
    '''
    def main(self):
        self.read_data_and_encode()
        trainx, testx = self.feature_selection()
        self.train_xgboost(self.train_X)
        print("Training is done.")
        predicted_labels_xg = self.predict(self.test_X)
        self.train_randforest(self.train_X)
        predicted_labels_randForest = self.predict(self.test_X)
        self.train_logreg(self.train_X)
        predicted_labels_logreg = self.predict(self.test_X)
        predicted_labels = []
        print("Prediction is done.")
        for i, j, k in zip(predicted_labels_xg, predicted_labels_randForest, predicted_labels_logreg):
            predicted_labels.append(self.mode([i, j, k]))

        result = DataFrame(column_stack((testx['orderItemID']+1, array(predicted_labels))), columns=['orderItemID', 'returnShipment'])
        result.to_csv('../result/submission_new.csv', index=False, sep=',', encoding='utf-8')

competitor = LfdCompetitor('../data/train.txt', '../data/test.txt')
competitor.main()
