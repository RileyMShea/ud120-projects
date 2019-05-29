#!/usr/bin/python

"""
    feature_format
    --------------
    A general tool for converting data from the
    dictionary format to an (n x k) python list that's 
    ready for training an sklearn algorithm

    n: count of k,v pairs in dict.items()
    k: count of features being extracted

    dictionary keys are names of persons in dataset
    dictionary values are dictionaries, where each
        key-value pair in the dict is the name
        of a feature, and its value for that person

    In addition to converting a dictionary to a numpy 
    array, you may want to separate the labels from the
    features--this is what targetFeatureSplit is for

    so, if you want to have the poi label as the target,
    and the features you want to use are the person's
    salary and bonus, here's what you would do:

    feature_list = ["poi", "salary", "bonus"] 
    data_array = feature_format( data_dictionary, feature_list )
    label, features = targetFeatureSplit(data_array)

    the line above (targetFeatureSplit) assumes that the
    label is the _first_ item in feature_list--very important
    that poi is listed first!
"""
import numpy as np


def feature_format(dictionary: dict,
                   features: list,
                   remove_nan=True,
                   remove_all_zeroes=True,
                   remove_any_zeroes=False,
                   sort_keys=False):
    """ convert dictionary to numpy array of features

        Parameters:
        -----------
        dictionary : dict
            dictionary.keys() : str
                Names of people in data set
            dictionary.values() : dict
                A nested dictionary for each person
                key: Feature
                value: Value for Feature
        features: list
            A list of features as strings.  These are the features that will be returned in the numpy array.
        remove_nan : Bool, default True
            Convert "NaN" string to 0.0
            To properly format all
        remove_all_zeroes : bool, default True
            Omit any data points for which all the features you seek are 0.0
        remove_any_zeroes : bool, default True
            Omit any data points for which any of the features you seek are 0.0
        sort_keys : bool, default True
            Sorts keys by alphabetical order. Setting the value as a string opens the corresponding pickle file with
            a preset key order (this is used for Python 3 compatibility, and sort_keys should be left as False for
            the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.

        Returns
        -----------
        np.array
            converted from 'dictionary' param
    """

    return_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.

    # if the sorts keys are an instance of str, ie
    if isinstance(object=sort_keys, # TODO : object/classinfo may break function
                  classinfo=str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            # replace 'NaN' strings with 0
            if value == "NaN" and remove_nan:
                value = 0
            tmp_list.append(float(value))

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        # if all features are zero and you want to remove
        # data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        # if any features for a given data point are zero
        # and you want to remove data points with any zeroes,
        # handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        # Append the data point if flagged for addition.
        if append:
            return_list.append(np.array(tmp_list))

    return np.array(return_list)


def targetFeatureSplit(data):
    """ 
        given a numpy array like the one returned from
        feature_format, separate out the first feature
        and put it into its own list (this should be the 
        quantity you want to predict)

        return targets and features as separate lists

        (sklearn can generally handle both lists and numpy arrays as 
        input formats when training/predicting)
    """

    target = []
    features = []
    for item in data:
        target.append(item[0])
        features.append(item[1:])

    return target, features
