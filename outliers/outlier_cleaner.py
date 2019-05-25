#!/usr/bin/python
from numpy import float64

def outlierCleaner(predictions: list,
                   ages: list,
                   net_worths: float64) -> list:
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Parameters:
        -----------
        predictions: list
            predicted targets from regression
        ages: list
            ages in the training set
        net_worths: float64
            actual value of net worths in the training set
        Returns:
        --------
        cleaned_data: list(*tuple)
            each tuple is of the form (age, net_worth, error)
            only the top 90%
    """

    cleaned_data = []

    # your code goes here
    zipped = zip(ages, net_worths, predictions)

    for entry in zipped:
        # residual error as absolute difference between net worth and prediction
        res_error = abs(entry[1] - entry[2])
        cleaned_data.append((entry[0], entry[1], res_error))

    # sort by residual order, slice highest 10% error rate off
    cleaned_data = sorted(cleaned_data, key=lambda x: x[2])[:int(.9 * len(ages))]

    return cleaned_data
