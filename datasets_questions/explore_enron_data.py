#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import numpy as np
import pandas as pd
import pickle
from tools.feature_format import featureFormat, targetFeatureSplit


enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
df = pd.DataFrame.from_dict(data=enron_data, orient='index')
# df.replace('NaN', 0)
# df_np = df.to_numpy()
# number of POI's in dataset
poi_num = sum([v["poi"] for c, v in enron_data.items() if v["poi"] == 1])

