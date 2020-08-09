import importlib
import pandas as pd
import numpy as np
import os
import csv
from csv import reader
from csv import writer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
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

# This function will take in the test features, and test outcome and output
# the competition formatted csv file to the filepath supplied.
# first we check if the Results directory exists, if not create it
# next we append the test results onto our test features
# finally we write each row in our newly appended test dataframe to the csv file
# this function supplies us with the test.csv file having the results of the test appended
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

# This function will return the k-resized training features and test features,
# first we create a random forest classifier as our feature importance classifier
# next we use the inbuilt feature_importances_ function that gives the list of features and a score relating to their importance
# we initialise a new dataframe using the top feature as our first new feature
# finally we add from the 2nd -> kth best feature columns from our original train and test set to our new feature selected return set
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

# this function will return the accuracy using 2 different techniques
# first method we will split the training set into an 80% training 20% validation set, whilst splitting at random
# next we will perform feature selection on our new test/validation set (see feature_select_topk)
# next we will create the same classifier we had used in our model, and fit the train/validaiton set
# finally we will compare the validation set results to the known results in our split

# second method we will use K-cross-fold validation using sci-kit package
# the method returns an array of scores from K-cross-fold validation
# we use this array to get the mean score, and the variance
def get_local_accuracy(features, target, neighbors):
    train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=50)
    selected_train_features, selected_test_features = feature_select_topk(train_features, train_target, test_features, 30)
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    classifier.fit(selected_train_features, train_target)
    test_results = classifier.predict(selected_test_features)
    accuracy = accuracy_score(test_results, test_target)
    scores = cross_val_score(classifier, features, target, cv = 5)
    print("Accuracy using 5 cross validaiton: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("The accuracy of KNN using train, test split " + str(accuracy*100) + "%")
        
def classify_knn(neighbors):
    train_features, train_target = get_train_data("../Data/train.csv")
    test_features = get_test_data("../Data/test.csv")
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    get_local_accuracy(train_features, train_target, neighbors)
    selected_train_features, selected_test_features = feature_select_topk(train_features, train_target, test_features, 30)
    classifier.fit(selected_train_features, train_target)
    test_result = classifier.predict(selected_test_features)
    save_test_results("Results/KNN_"+str(neighbors)+".csv", test_features, test_result)


classify_knn(243)
classify_knn(3000)

for i in range(100, 3100, 100):
    classify_knn(i)