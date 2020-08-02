import pandas as pd
import numpy as np
import os
import csv
from csv import reader
from csv import writer
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def process_train_data(filepath):
    entries = pd.read_csv(filepath)
    # remove any row with missing fields
    entries.dropna(axis=0, how='any', inplace=True)
    features = entries.iloc[:, :-1]
    target = entries.iloc[:, -1]
    return features, target

def process_test_data(filepath):
    entries = pd.read_csv(filepath)
    # remove any row with missing fields
    entries.dropna(axis=0, how='any', inplace=True)
    features = entries.iloc[:, :]
    return features

def append_test_result(filepath, test_features, test_results):
    # get all test data and append to end the result
    rows = np.shape(test_features)[0]
    cols = np.shape(test_features)[1]
    frame = pd.DataFrame(test_features)

    frame['Class'] = test_results
    if not os.path.exists('Results'):
        os.makedirs('Results')
    with open(filepath, 'w+') as file:
        writer = csv.writer(file)
        for i in range(rows):
            writer.writerow(frame.iloc[i])




def classify_knn(neighbors):
    train_features, train_target = process_train_data("../Data/train.csv")
    test_features = process_test_data("../Data/test.csv")
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    classifier.fit(train_features, train_target)
    test_result = classifier.predict(test_features)
    append_test_result("Results/KNN_"+str(neighbors)+".csv", test_features, test_result)


classify_knn(5)