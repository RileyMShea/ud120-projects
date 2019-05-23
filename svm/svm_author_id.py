#!/usr/bin/python
from tools.email_preprocess import preprocess
import numpy as np
from time import time
from sklearn import svm
from sklearn.metrics import accuracy_score

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
# your code goes here ###

# use subset of full dataset?
subset = False

if subset:
    features_train = features_train[:len(features_train) // 100]
    labels_train = labels_train[:len(labels_train) // 100]

clf = svm.SVC(kernel='rbf', C=10000)


t0 = time()
# Train the classifier
clf.fit(features_train, labels_train)
print("Training time:", round(time()-t0, 3), "s")

t1 = time()
# Make predictions with our classifier
pred = clf.predict(features_test)
answer = pred[100]
print(f"Prediction: {answer}")
print("Prediction time:", round(time()-t1, 3), "s")
# How accurate are our predictions
accuracy = accuracy_score(pred, labels_test)
print(f"Accuracy: {accuracy*100:.2f}%")

#########################################################
