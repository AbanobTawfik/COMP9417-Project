import importlib
import pandas as pd
import numpy as np
import os
import csv
from csv import reader
from csv import writer
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def get_train_data(filepath):
    entries = pd.read_csv(filepath)
    # remove any row with missing fields
    entries.dropna(axis=0, how='any', inplace=True)
    features = entries.iloc[:, 1:-1]
    target = entries.iloc[:, -1]

    return features, target

def get_test_data(filepath):
    entries = pd.read_csv(filepath)
    # remove any row with missing fields
    entries.dropna(axis=0, how='any', inplace=True)
    features = entries.iloc[:, :]
    return features

def save_test_results(filepath, test_features, test_results):
    # get all test data and append to end the result
    rows = np.shape(test_features)[0]
    cols = np.shape(test_features)[1]
    frame = pd.DataFrame(test_features)

    fields = ['id']
    for i in range(9):
        fields.append(f'Class_{i+1}')

    row_template = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    frame['Class'] = test_results
    if not os.path.exists('Results'):
        os.makedirs('Results')
    with open(filepath, 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for i, row in enumerate(test_results):
            out_row = row_template.copy()
            out_row[int(row[-1]) - 1] = 1
            writer.writerow([i + 1] + out_row)

def feature_select_topk(train_features, train_target, test_features, k):
    feature_importance_classifier = RandomForestClassifier(n_estimators=100)
    feature_importance_classifier.fit(train_features, train_target)
    imp_feature = feature_importance_classifier.feature_importances_
    indices = np.argsort(imp_feature)[::-1]
    new_train_features = pd.DataFrame(train_features.iloc[:,indices[0]])
    new_test_features = pd.DataFrame(test_features.iloc[:,indices[0]])
    for i in range (1, k):
        train_feature = train_features.iloc[:,indices[i]]
        new_train_features = new_train_features.join(train_feature)
        test_feature = test_features.iloc[:, indices[i]]
        new_test_features = new_test_features.join(test_feature)

    return new_train_features, new_test_features

        

def classify_knn(neighbors):
    train_features, train_target = get_train_data("../Data/train.csv")
    test_features = get_test_data("../Data/test.csv")
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    selected_train_features, selected_test_features = feature_select_topk(train_features, train_target, test_features, 20)
    classifier.fit(selected_train_features, train_target)
    test_result = classifier.predict(selected_test_features)
    save_test_results("Results/KNN_"+str(neighbors)+".csv", test_features, test_result)


classify_knn(244)
# classify_knn(3000)

# for i in range(100, 3100, 100):
#     classify_knn(i)