# %% Imports
# !/usr/bin/python
import pickle  # For import/export of python objects to a file
import os  # to set cwd
import sys  # to add paths to access local user packages/scripts
import numpy as np   # to format and process data for sklearn
import pandas as pd  # to reshape and explore data, features selection
from pandas.plotting import scatter_matrix  # a special pandas method for pairwise scatterplot grid
import seaborn as sns # For enhanced visualizations and additional plotting functions
import matplotlib.pyplot as plt # primary plotting library
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pprint import pprint
from tools.feature_format import feature_format, targetFeatureSplit
from final_project.tester import dump_classifier_and_data

os.chdir("./final_project")  # TODO: remove in final submission
# %% Importing Dataset from pickled dictionary
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
# replace 'Nan' strings with None in dataset
for outer_keys, inner_dicts in data_dict.items():
    for k, v in inner_dicts.items():
        if v == 'NaN':
            data_dict[outer_keys][k] = None

df = pd.DataFrame.from_dict(data_dict, orient='index')

df['has_email'] = df.email_address != "NaN"
df.pop('email_address')

# %% Kitchen Sink attempt
features = df[['salary', 'bonus', 'poi']].dropna()
labels = features['poi']  # The target variable
features.pop('poi')

features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                            test_size=.3,
                                                                            random_state=42)

try:
    clf = LogisticRegression().fit(features_train, labels_train)
except ValueError as e:
    print(e)
# %% testing model
pred = clf.predict(features_test)
# %% plot test
# sns.lmplot('salary', 'bonus', features_test)
features = df[['salary', 'bonus', 'poi']].dropna()

sns.lmplot('salary', 'bonus', features, hue='poi')
plt.show()
# %%
from choose_your_own.class_vis import pretty_picture
pretty_picture(clf, features_test, labels_test)
