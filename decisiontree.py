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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image 
import graphviz

train_otto = pd.read_csv('train.csv')
test_otto = pd.read_csv('test.csv')

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
odt = odt.fit(train_x, train_y)

y_pred = odt.predict_proba(test_x)

fields = ['id']
for i in range(1,10):
    fields.append(f'Class_{i}')
with open(f'./decisiontree_pred.csv', mode='w', newline='') as f:
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
orf.fit(train_x, train_y)
y_pred_rf = orf.predict_proba(test_x)

fields = ['id']
for i in range(1,10):
    fields.append(f'Class_{i}')
with open(f'./decisiontree_pred_rf.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    for i, row in enumerate(y_pred_rf):
        entry = [int(i + 1)] + [val for val in row]
        writer.writerow(entry)




