#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

### Decision tree 
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print "accuracy is:", accuracy_score(labels_test, pred)

print "Number of poi's in the test set is:", sum(pred)
print "Number of people in the test set is:", len(features_test)
print "Precision of the POI identifier is:", precision_score(labels_test, pred)
print "Recall of the POI identifier is:", recall_score(labels_test, pred)

