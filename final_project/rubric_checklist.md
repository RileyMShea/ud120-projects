### Data Exploration
# total number of data points
146 y
allocation across classes (POI/non-POI)
number of features used
are there features with many missing values? etc.


#%% md
## Data Exploration

We are working with three effective data sources:
* The CALO Enron Email Dataset
    * A Cleaned set of emails between enron Emails
* Enron Statement of financial affairs
    * Document detailing financial features of Enron employees
* An Associated Press article from 2005 Listing POI's.  Poi's are subcategorized in the article as:
    * Newly Convicted
    * On Trial
    * Awaiting Trial
    * Previously Convicted at Trial
    * Acquitted at Trial
    * Conviction Overturned
    * Guilty Pleas
    * Settlements  


### Issues

* NaN strings affecting column datatype in pandas
    * Handled by setting all 'NaN' strings to None before importing data_dict into pandas
    dataframe
* Email_Address field, upon import, is not useful in ML classification
    * All emails are either: NaN or an email from the 'enron.com' domain.
    * Potentially usable if remapped to a Binary 'Has_email' Feature, based on whether the
    email is present or 'NaN'
* Aggregate Features: Total Payments, Total Stock Value
    * Computed features
* Aggregate Record: TOTAL
    * A record was imported that was the aggregate total for each feature.  This would
    heavily skew data and negatively impact classification accuracy
* Non-person entity record: 'THE TRAVEL AGENCY IN THE PARK'
    * Found to be owned by sister of Enron Chairperson. As we are trying to predict persons
    of interest, this field will be removed since it is a company.
* Person with all NaN Features
    * Record with index "LOWRY CHARLES P" has all nan values for every feature(except poi).
    A google of search of 'enron charles p lowry" is yields little information, thus record
    is being dropped. 
* All POI in the dataset have at least one non-value feature value from both the Financial
and 
* 'Person of Interest' seems somewhat vague.
> "Person of interest" is a term used by U.S. law enforcement when identifying someone involved
in a criminal investigation who has not been arrested or formally accused of a crime. It has no legal
meaning, but refers to someone in whom the police are "interested," either because the person is cooperating
with the investigation, may have information that would assist the investigation, or possesses certain
characteristics that merit further attention.

    * This means we aren't necessarily predicting whether someone is guilty, only if they are
     Case could be made to changed Person of interest from a Binary feature to a 
    categorical one using the statuses from the AP article from 2005.

## Features

There is a large number of NaN values in this dataset.  This is partly due to Email data being merged with financial
data.  

### Email Features
* email_address
    * Many NaN values:  All records with a NaN email address had no emails inside the CALO Enron Email Dataset.
    
    

### Payment Features

* Salary: int
    > Reflects items such as base salary, executive cash allowances, and benefits payments.
* Bonus
    > Reflects annual cash incentives paid based upon company performance. Also may include other retention payments.
* Long Term Incentive: int
    > Reflects long-term incentive cash payments from various long-term incentive programs designed to tie executive compensation to long-term success as measured
against key performance drivers and business objectives over a multi-year period, generally 3 to 5 years.
* Deferred Income: int
    > Reflects voluntary executive deferrals of salary, annual cash incentives, and long-term cash incentives as well as cash fees deferred by non-employee directors
under a deferred compensation arrangement. May also reflect deferrals under a stock option or phantom stock unit in lieu of cash arrangement.
* Deferral Payments: int
    > Reflects distributions from a deferred compensation arrangement due to termination of employment or due to in-service withdrawals as per plan provisions.
* Loan Advances: int
    > Reflects total amount of loan advances, excluding repayments, provided by the Debtor in return for a promise of repayment. In certain instances, the terms of the
promissory notes allow for the option to repay with stock of the company.
* Other
    > Reflects items such as payments for severence, consulting services, relocation costs, tax advances and allowances for employees on international assignment (i.e.
housing allowances, cost of living allowances, payments under Enron’s Tax Equalization Program, etc.). May also include payments provided with respect to
employment agreements, as well as imputed income amounts for such things as use of corporate aircraft.
* Expenses
    > Reflects reimbursements of business expenses. May include fees paid for consulting services.
* Director Fees
    >  Reflects cash payments and/or value of stock grants made in lieu of cash payments to non-employee directors.

### Stock Value Features
> In 1998, 1999 and 2000, Debtor and non-debtor affiliates were charged for options granted. The Black-Scholes method was used to determine the amount to be
charged. Any amounts charged to Debtor and non-debtor affiliates associated with the options exercised related to these three years have not been subtracted
from the share value amounts shown.

* Exercised Stock Options
    > Reflects amounts from exercised stock options which equal the market value in excess of the exercise price on the date the options were exercised either through
cashless (same-day sale), stock swap or cash exercises. The reflected gain may differ from that realized by the insider due to fluctuations in the market price and
the timing of any subsequent sale of the securities.
* Restricted Stock Options
    > Reflects the gross fair market value of shares and accrued dividends (and/or phantom units and dividend equivalents) on the date of release due to lapse of vesting
periods, regardless of whether deferred.
* Restricted Stock Deferred
    > Reflects value of restricted stock voluntarily deferred prior to release under a deferred compensation arrangement.


* Want to find outliers in the form of POI but not other outliers
* We know from the mini-projects that the index entry name 'TOTAL' is the wrong kind of outlier that we want
* Manual inspection of index names also revealed the existence of 'THE TRAVEL AGENCY IN THE PARK"
    * Seems to be the travel agency of choice for Enron employees.  Essentially mandatory to use for Enron employees.
    * Not a 'person' and not an Enron employee.  Also many NaN fields including Email Address.  Dropping this col.
