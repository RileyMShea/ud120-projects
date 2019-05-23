#!/usr/bin/python
from tools.email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score
from time import time

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################

clf = tree.DecisionTreeClassifier(min_samples_split=40)

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

#########################################################
