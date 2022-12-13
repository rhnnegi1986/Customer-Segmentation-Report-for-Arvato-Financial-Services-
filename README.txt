                                                    Project Title

Customer Segmentation Report for Arvato Financial Solutions



Modules and libraries installation
In order to successfully run the packages we need to pip install the packages and restart the kernal before importing these libraries in our notebook.

! python -m pip install experimental
! python -m pip install sklearn_evaluation
! python -m pip install missingno
! python -m pip install scikit-learn --user --upgrade pip

libraries
import warnings
import missingno as msno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import  IterativeImputer
from sklearn import linear_model
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn_evaluation import plot
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import metrics
from keras import regularizers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier





Business Understanding
Arvato Funancials Services is looking to send out a mail-order to the target segment of population in Germany which are extreamly likely to become company's customers.

Introduction

This project is based on analyzing demographics data of customers for the mail-order company based in Germany.Projects scope consists of utilizing Unsupervised and Supervised machine learning techniques 
in performing customer segmentation, identifying the parts of the population that best describe the core customer base of the company,implementing and comparing the the analysis of customer segementation on 
Germany's general population.Further this learning would be implemented on the third demographic dataset that identified as target segament for the campaign for the company with high probability of becoming the potentential clients for Arvato Financials.


Key Questions for the project
1. What are the key independent variables in current customer demographic that are highly correlated with the target variable (Online Purchase)
2. What are the independent variables that are highly correlated to each other
3. How will statistically important independent variables from current customers perform against Germany's General Population.
4. Finaly, we will be making predictions on the test dataset to target the customers who are likely to become company's customer.

There are four data files associated with this project:

Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).
Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. Use the information from the first two files to figure out how customers ("CUSTOMERS") are similar to or differ from the general population at large ("AZDIAS"), then use your analysis to make predictions on the other two files ("MAILOUT"), predicting which recipients are most likely to become a customer for the mail-order company.

The "CUSTOMERS" file contains three extra columns ('CUSTOMER_GROUP', 'ONLINE_PURCHASE', and 'PRODUCT_GROUP'), which provide broad information about the customers depicted in the file. The original "MAILOUT" file included one additional column, "RESPONSE", which indicated whether or not each recipient became a customer of the company. For the "TRAIN" subset, this column has been retained, but in the "TEST" subset it has been removed; it is against that withheld column that your final predictions will be assessed in the Kaggle competition.

Otherwise, all of the remaining columns are the same between the three data files. For more information about the columns depicted in the files, you can refer to two Excel spreadsheets provided in the workspace. [One of them](./DIAS Information Levels - Attributes 2017.xlsx) is a top-level list of attributes and descriptions, organized by informational category. [The other](./DIAS Attributes - Values 2017.xlsx) is a detailed mapping of data values for each feature in alphabetical order.

In the below cell, we've provided some initial code to load in the first two datasets. Note for all of the .csv data files in this project that they're semicolon (;) delimited, so an additional argument in the read_csv() call has been included to read in the data properly. Also, considering the size of the datasets, it may take some time for them to load completely.

You'll notice when the data is loaded in that a warning message will immediately pop up. Before you really start digging into the modeling and analysis, you're going to need to perform some cleaning. Take some time to browse the structure of the data and look over the informational spreadsheets to understand the data values. Make some decisions on which features to keep, which features to drop, and if any revisions need to be made on data formats. It'll be a good idea to create a function with pre-processing steps, since you'll need to clean all of the datasets before you work with them.
Projects Road Map
This project undertook Unsupervised Machine Learning and Supervised Machine Learning approach:
1.Unsupervised Learning
  1a. Descriptive Analytics- In this section we performed data extraction, data manipulation and data visualization to identify the independent and target variable
  -- insert photoes
  1b. Prescriptive Analytics- In this section we performed feature selection process to identify and prescribe independent variables which are statistically important to our target variable
  -- inser photoes 


2.Supervised Learning
  2a.  Descriptive Analytics- From the analysis performed Unsupervised Learning,in this section we performed extended and implemented Supervised learning algorithim in data extraction, data manipulation 
       to identify the independent and target variable
  -- insert photoes
  2b. Prescriptive Analytics- In this section we performed feature selection process to identify and prescribe independent variables which are statistically important to our target variable

 2c. Predictive Analytics - Combining all the lessons leaned through supervised and unsupervised learning , we then were able to predict our target variable on the provided test data  




                                               Final Analysis

The project was able to predict the list of Features that have the potential of becoming the prospective customers of Arvato Financials Services. There are number of takeaways that can be derived from the above output window. From the analysis we have conducted we can predict that there are 652 individuals in the maiout-test data which can be made Arvato Financials Services prospective customers.From the output window as above there are certain class with those indiviual population which are more likely to become Arvato Financials Services customers from others. 

The final analysis was able to identify and predict Germany's general population demographics. Some of the major takeaways from this were:

1. Age Type - Germany's Culturely Elederly population is more like to become Arvato Financials Services prospective customers.
2. Buyer Type - Customers more than 2 years with the Arvato Financials Services are more likely to buy another product.
3. Customer journey typology- Individuals who are seeking orientation are more likely to become customers than their counterpart.
4. TECHNOLOGY - Here our analyis predicted that Individuals who have indifferent mindset with respect to Technology product group are likely to become Arvato Financials Services prospective customers.
5. Mail-Order Online/Mail-Order Online - Arvato Financials Services Customers with higest online/offline activity in last 3 years are likely to buy another finanacial product with the company.
6. BOOKS and CDS - BOOKS and CDS Buyers > 24 months of purchasing history are more likely to become Arvato Financials Services Customers.
7. HOUSE DECORATION - Individuals with who have made a purchase of HOUSE DECORATION Buyers > 24 months or higher are likely to become customers.
8. TOTAL - Individuals with 100% Online-transactions within the last 12 months are extreamly likely to become Arvato Financials Services customers.
9. DIETARY SUPPLEMENTS - Our Analysis was able to predict that individuals who have a transactional activity in DIETARY SUPPLEMENTS for 24 months or higher are likely to become Arvato Financials Services customer as well.
10. ALL OTHER CATEGORIES - Finally our analysis also predicted that individuals whose transational activity is extreamly high in All Other product Categories are very likely to become customers of Arvato Financials Services as well.   







