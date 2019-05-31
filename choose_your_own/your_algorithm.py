#!/usr/bin/python

import matplotlib.pyplot as plt
from choose_your_own.prep_terrain_data import make_terrain_data
from choose_your_own.class_vis import pretty_picture
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from time import time

features_train, labels_train, features_test, labels_test = make_terrain_data()

# the training data (features_train, labels_train) have both "fast" and "slow"
# points mixed together--separate them so we can give them different colors
# in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

# initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


# your code here!  name your classifier object clf if you want the
# visualization code (pretty_picture) to show you the decision boundary
clf = RandomForestClassifier(n_estimators=1000,
                             criterion="gini",
                             max_depth=None,
                             min_samples_split=8,
                             min_samples_leaf=1,
                             min_weight_fraction_leaf=0,
                             max_features='auto',
                             max_leaf_nodes=None,
                             min_impurity_decrease=0,
                             bootstrap=True,
                             oob_score=False,
                             n_jobs=None,
                             random_state=None,
                             verbose=0,
                             warm_start=False,
                             class_weight=None
                             )

t0 = time()
# Train the classifier
clf.fit(features_train, labels_train)
print("Training time:", round(time() - t0, 3), "s")

t1 = time()
# Make predictions with our classifier
pred = clf.predict(features_test)
print("Prediction time:", round(time() - t1, 3), "s")
# How accurate are our predictions
accuracy = accuracy_score(pred, labels_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

try:
    # Save plot to file
    pretty_picture(clf, features_test, labels_test)
except NameError as ne:
    print("NameError while trying to use pretty picture")
    print(ne)
