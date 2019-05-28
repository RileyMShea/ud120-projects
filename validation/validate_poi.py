#!/usr/bin/python
# core libraries
import pickle
from time import time
# sci-kit learn imports
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# local package imports
from tools.feature_format import feature_format, targetFeatureSplit


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

# first element is our labels, any added elements are predictor
# features. Keep this the same for the mini-project, but you'll
# have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = feature_format(data_dict, features_list, sort_keys='../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)

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
