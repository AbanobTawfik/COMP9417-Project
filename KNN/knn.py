import pandas as pd
import numpy as np
import os
import csv
from csv import reader
from csv import writer
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def get_train_data(filepath):
    entries = pd.read_csv(filepath)
    # remove any row with missing fields
    entries.dropna(axis=0, how='any', inplace=True)
    features = entries.iloc[:, :-1]
    target = entries.iloc[:, -1]

    return features, target

def get_test_data(filepath):
    entries = pd.read_csv(filepath)
    # remove any row with missing fields
    entries.dropna(axis=0, how='any', inplace=True)
    features = entries.iloc[:, :]
    return features

<<<<<<< HEAD
def create_test_result_output_file(filepath, test_features, test_results):
=======
def save_test_results(filepath, test_features, test_results):
>>>>>>> da6e8fe34496d60ec0fad03779d3cedd2ff0dae0
    # get all test data and append to end the result
    rows = np.shape(test_features)[0]
    cols = np.shape(test_features)[1]
    frame = pd.DataFrame(test_features)

    fields = ['id']
    for i in range(10):
        fields.append(f'Class_{i}')

    row_template = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    frame['Class'] = test_results
    if not os.path.exists('Results'):
        os.makedirs('Results')
    with open(filepath, 'w+') as file:
        writer = csv.writer(file)
<<<<<<< HEAD
        for i in range(rows):
            writer.writerow(frame.iloc[i])

def create_submit_file(model, file_path, test_features):
    y_pred = model.predict_proba(test_features)
    y_pred = np.vectorize(lambda x: round(x, 4))(y_pred)
    # Write results to csv file
    fields = ['id']
    if not os.path.exists('Results'):
        os.makedirs('Results')
    for i in range(1,10):
        fields.append('Class_' + str(i))
    with open(file_path, mode='w+') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i, row in enumerate(y_pred):
            entry = [int(i + 1)] + [val for val in row]
            writer.writerow(entry)


=======
        writer.writerow(fields)
        for i, row in enumerate(test_results):
            out_row = row_template.copy()
            out_row[int(row[-1]) - 1] = 1
            writer.writerow([i + 1] + out_row)
>>>>>>> da6e8fe34496d60ec0fad03779d3cedd2ff0dae0

def classify_knn(neighbors):
    train_features, train_target = get_train_data("../Data/train.csv")
    test_features = get_test_data("../Data/test.csv")
<<<<<<< HEAD
    classifier = KNeighborsClassifier(n_neighbors=neighbors, p=3)
    classifier.fit(train_features, train_target)
    test_result = classifier.predict(test_features)
    # create_test_result_output_file("Results/KNN_"+str(neighbors)+".csv", test_features, test_result)
    create_submit_file(classifier, "Results/KNN_"+str(neighbors)+".csv", test_features)

classify_knn(10)
=======
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    classifier.fit(train_features, train_target)
    test_result = classifier.predict(test_features)
    save_test_results("Results/KNN_"+str(neighbors)+".csv", test_features, test_result)


classify_knn(5)
>>>>>>> da6e8fe34496d60ec0fad03779d3cedd2ff0dae0
