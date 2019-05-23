#!/usr/bin/python
from tools.email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from time import time

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""


# sys.path.append("../tools/")

# features_train and features_test are the features for the training
# and testing dataset, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
# your code goes here ###


clf = GaussianNB()

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
