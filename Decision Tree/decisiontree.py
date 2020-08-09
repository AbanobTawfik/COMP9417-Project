import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn import metrics
import csv
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image 
import graphviz

if not os.path.exists('Results'):
    os.makedirs('Results')

# see KNN.py in the 'KNN' folder for a description of this function
def get_local_accuracy(features, target, model):
    train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=50)
    model.fit(train_features, train_target)
    test_results = model.predict(test_features)
    accuracy = accuracy_score(test_results, test_target)
    scores = cross_val_score(model, features, target, cv = 5)
    print("Accuracy using 5 cross validaiton: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("The accuracy of KNN using train, test split " + str(accuracy*100) + "%")

train_otto = pd.read_csv('../Data/train.csv')
test_otto = pd.read_csv('../Data/test.csv')

train = train_otto.to_numpy()[:, 1:]
test_x = test_otto.to_numpy()[:, 1:]

# Convert target column to ints
# (it is initially strings of the form 'Class_i')
for row in train:
    target = int(row[-1][-1])
    row[-1] = target

np.random.shuffle(train)
train_x = train[:, :-1].astype(int)
train_y = train[:, -1].astype(int)

# Creating decision tree classifier
odt = tree.DecisionTreeClassifier()
get_local_accuracy(train_x, train_y, odt)
odt = odt.fit(train_x, train_y)

y_pred = odt.predict_proba(test_x)

fields = ['id']
for i in range(1,10):
    fields.append(f'Class_{i}')
with open(f'Results/decisiontree_pred.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    for i, row in enumerate(y_pred):
        entry = [int(i + 1)] + [val for val in row]
        writer.writerow(entry)
     

# Visualising decision tree
features=[]
for i in range(1, 94):
    features.append(f'feat_{i}')
    
classes=[]
for i in range(1, 10):
  classes.append(f'Class_{i}')

dot_data = tree.export_graphviz(odt, out_file=None,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features,class_names=classes)

graph = graphviz.Source(dot_data)
graph



# Creating random forest classifier
orf = RandomForestClassifier(n_estimators=100)
get_local_accuracy(train_x, train_y, orf)
orf.fit(train_x, train_y)
y_pred_rf = orf.predict_proba(test_x)

fields = ['id']
for i in range(1,10):
    fields.append(f'Class_{i}')
with open(f'Results/decisiontree_pred_rf.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    for i, row in enumerate(y_pred_rf):
        entry = [int(i + 1)] + [val for val in row]
        writer.writerow(entry)