#!/usr/bin/python

import pickle
import matplotlib.pyplot as plt
from tools.feature_format import feature_format, targetFeatureSplit

# read in data dictionary, convert to numpy array
# data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
data_dict = pickle.load(open("../final_project/final_project_dataset_modified.pkl", "rb"))
# del data_dict['TOTAL']
features = ["salary", "bonus"]
data = feature_format(data_dict, features, sort_keys=True)
# your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
