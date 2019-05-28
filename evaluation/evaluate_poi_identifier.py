#!/usr/bin/python
# core libraries
import pickle
from time import time
# sci-kit learn imports
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
# local package imports
from tools.feature_format import feature_format, targetFeatureSplit

"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

# TODO: add more features to features_list!
features_list = ["poi", "salary"]

data = feature_format(data_dict, features_list, sort_keys='../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)

# TODO: your code goes here
features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                           test_size=.3,
                                                                           random_state=42)
# it's all yours from here forward!

clf = tree.DecisionTreeClassifier()

t0 = time()
# Train the classifier
clf.fit(features_train, labels_train)
print("Training time:", round(time()-t0, 3), "s")

t1 = time()
# Make predictions with our classifier
pred = clf.predict(features_test)
print("Prediction time:", round(time()-t1, 3), "s")
# How accurate are our predictions
accuracy = accuracy_score(pred, labels_test)
print(f"Accuracy: {accuracy*100:.2f}%")
