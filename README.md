Machine Learning: Final Project
==============

###Goal:
Build POI(Person of Interest Identifier) from:

* financial data
* email data

True POI's: Individuals that were found guilty

> A hand generated list of these true POI's has been provided.

## Files to get started

### poi_id.py

Starter code for the POI identifier, you will write your analysis here.
You will also submit a version of this file for your evaluator to verify your algorithm and results. 

### final_project_dataset.pkl

The dataset for the project, more details below. 

### tester.py

When you turn in your analysis for evaluation by Udacity,
you will submit the algorithm, dataset and list of features that you use (these are created automatically in poi_id.py).
The evaluator will then use this code to test your result,
to make sure we see performance that’s similar to what you report.
You don’t need to do anything with this code,
but we provide it for transparency and for your reference. 

### Directory: 'emails_by_address'

this directory contains many text files,
each of which contains all the messages to or from a particular email address.
It is for your reference, if you want to create more advanced features based on the details of the emails dataset.
You do not need to process the e-mail corpus in order to complete the project

## My Job

* engineer the features
* pick and tune an algorithm
* test and evaluate identifier

## Data format

Enron Email and finance data in a dictionary.  Each k, v is a person

financial features = ['salary', 'deferral_payments',
'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] 

Units in US dollars

email features: ['to_messages', 'email_address', 'from_poi_to_this_person',
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']


Units in # of email messages

POI label: [‘poi’]

(boolean, represented as integer)

Encouraged to:

* Make, transform/rescale new features from starter features(**at least one required**)
    * new features should be added to my_dataset and my_feature_list


1. Summarize for us the goal of this project and how machine learning is
useful in trying to accomplish it. As part of your answer, give some
background on the dataset and how it can be used to answer the project
question. Were there any outliers in the data when you got it, and how did
you handle those? [relevant rubric items: “data exploration”, “outlier
investigation”]

```python
s = "Python syntax highlighting"
print s
```

> Goal of this project is to take a number of input features, then create a classifier with machine
learning, and use it to predict whether a person is a person of interest(POI).  Goal to maximize F1 as a measure of
precision and recall.

    Maximizing F1 will be done by data cleaning, Feature selection, classifier selection, and classifer tuning
 
### What features did you end up using in your POI identifier, and what
selection process did you use to pick them?

### Did you have to do any scaling? Why or why not?

As part of the assignment, you should attempt
to engineer your own feature that does not come ready-made in the
dataset -- explain what feature you tried to make, and the rationale behind
it. (You do not necessarily have to use it in the final analysis, only engineer
and test it.) In your feature selection step, if you used an algorithm like a
decision tree, please also give the feature importances of the features that
you use, and if you used an automated feature selection function like
SelectKBest, please report the feature scores and reasons for your choice
of parameter values. [relevant rubric items: “create new features”,
“intelligently select features”, “properly scale features”]




