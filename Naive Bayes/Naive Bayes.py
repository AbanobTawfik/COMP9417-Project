import pandas as pd
import numpy as np
import os
import csv
from csv import reader
from csv import writer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
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

def create_test_result_output_file(filepath, test_features, test_results):
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

def get_local_accuracy(features, target, model, name):
    train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=50)
    selected_train_features, selected_test_features = feature_select_topk(train_features, train_target, test_features, 30)
    classifier = model
    classifier.fit(selected_train_features, train_target)
    test_results = classifier.predict(selected_test_features)
    accuracy = accuracy_score(test_results, test_target)
    print("The accuracy of " + name + " was " + str(accuracy*100) + "%")


def classify_nb(model, filename):
    train_features, train_target = get_train_data("../Data/train.csv")
    test_features = get_test_data("../Data/test.csv")
    selected_train_features, selected_test_features = feature_select_topk(train_features, train_target, test_features, 20)
    classifier = model
    get_local_accuracy(train_features, train_target, model, filename)
    classifier.fit(selected_train_features, train_target)
    test_result = classifier.predict(selected_test_features)
    print(test_result)
    # create_test_result_output_file("Results/Naive_Bayes_"+str(neighbors)+".csv", test_features, test_result)
    create_submit_file(classifier, "Results/"+filename+".csv", selected_test_features)

gnb = GaussianNB()
bnb = BernoulliNB()
mnb = MultinomialNB()

classify_nb(gnb, "Guassian_Naive_Bayes")
classify_nb(bnb, "Bernoulli_Naive_Bayes")
classify_nb(mnb, "Multinomial_Naive_Bayes")
