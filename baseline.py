'''
Train a logistic regression model on the Otto Group dataset and write the results
of running the model on test data to a csv file.
This baseline model receives a final score of log_loss = 0.674 when run on the
whole test dataset
'''

import os
import numpy as np
import pandas
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

if os.path.isdir('Data'):
    data_path = 'Data'
else:
    data_path = '.'

# Get data and remove id column
labelled = pandas.read_csv(f'{data_path}/train.csv').to_numpy()[:, 1:]
test_x = pandas.read_csv(f'{data_path}/test.csv').to_numpy()[:, 1:]

# Convert target column to ints
# (it is initially strings of the form 'Class_i')
for row in labelled:
    target = int(row[-1][-1])
    row[-1] = target

np.random.shuffle(labelled)
train_x = labelled[:, :-1].astype(int)
train_y = labelled[:, -1].astype(int)

model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(train_x, train_y)

y_pred = model.predict_proba(test_x)
y_pred = np.vectorize(lambda x: round(x, 4))(y_pred)


def get_local_accuracy(features, target):
    train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=50)
    classifier = LogisticRegression()
    classifier.fit(train_features, train_target)
    test_results = classifier.predict(test_features)
    accuracy = accuracy_score(test_results, test_target)
    scores = cross_val_score(classifier, features, target, cv = 5)
    print("Accuracy using 5 cross validaiton: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("The accuracy of KNN using train, test split " + str(accuracy*100) + "%")

get_local_accuracy(train_x, train_y)

# Write results to csv file
fields = ['id']
for i in range(1,10):
    fields.append(f'Class_{i}')
with open(f'{data_path}/baseline_pred.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    for i, row in enumerate(y_pred):
        entry = [int(i + 1)] + [val for val in row]
        writer.writerow(entry)
