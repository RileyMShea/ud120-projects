# %% Imports
# !/usr/bin/python
import pickle
import os
import sys
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from pprint import pprint
from tools.feature_format import feature_format, targetFeatureSplit
from final_project.tester import dump_classifier_and_data

os.chdir("./final_project")  # TODO: remove in final submission
# %% Importing Dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
df = pd.DataFrame.from_dict(data_dict, orient='index')
# %% Kitchen Sink attempt
train = df
y_train = train['poi']  # The target variable
poi_ser = train.pop('poi')
try:
    modelFit = LogisticRegression().fit(train, y_train)
except ValueError as e:
    print(e)
    pd.concat(train, poi_ser)
