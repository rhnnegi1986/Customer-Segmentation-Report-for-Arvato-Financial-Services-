                                                    Project Title

Customer Segmentation Report for Arvato Financial Solutions



Modules and libraries installation
In order to successfully run the packages we need to pip install the packages and restart the kernal before importing these libraries in our notebook.

! python -m pip install experimental
! python -m pip install sklearn_evaluation
! python -m pip install missingno
! python -m pip install scikit-learn --user --upgrade pip

libraries
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import missingno as msno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import metrics
from keras import regularizers
from sklearn.feature_selection import SelectKBest,SelectFromModel
from sklearn.preprocessing import MinMaxScaler,OrdinalEncoder,LabelEncoder,StandardScaler
from sklearn.compose import make_column_selector as selector,ColumnTransformer
from sklearn_evaluation import plot
from sklearn.inspection import permutation_importance
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import  IterativeImputer
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier,VotingClassifier,ExtraTreesClassifier,RandomForestClassifier

%matplotlib inline




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
File Repository

Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).

Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).

Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).

Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).

DIAS Information Levels - Attributes 2017.xlsx : This is a top-level list of attributes and descriptions, organized by informational category.

DIAS Attributes - Values 2017.xlsx: This is a detailed mapping of data values for each feature in alphabetical order.

Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood.

The "CUSTOMERS" file contains three extra columns ('CUSTOMER_GROUP', 'ONLINE_PURCHASE', and 'PRODUCT_GROUP'), which provide broad information about the customers depicted in the file. 
The original "MAILOUT" file included one additional column, "RESPONSE", which indicated whether or not each recipient became a customer of the company. 



Steps thats taken for the successful completion of this project:

Data Preprocessing — Data Cleansing, Missing Values identification, Missing values Imputation

Feature Selection using Unsupervised Machine Learning — Involving dimensionality reduction technique like Random Forest Classifier

Customer Segmentations — Permutation model for Feature Importance using Supervised Learning

Model Selection deployment and Prediction with Supervised Machine Learning Models — Stochastic Gradient Descent, Support Vector Machines, Decision Trees, Random Forest, Bernoulli NB, K-Nearest Neighbor and Bagging Classifier
                                            


Conclusion
The project was able to predict the list of Features that have the potential of becoming the prospective customers of Arvato Financials Services. 
There are number of takeaways that can be derived from the above output window. From the analysis we have conducted we can predict that there are 652 individuals in the maiout-test data which can be made Arvato Financials Services prospective customers.From the output window as above there are certain class with those indiviual population which are more likely to become Arvato Financials Services customers from others. 

The final analysis was able to identify and predict Germany’s general population demographics. Some of the major takeaways from this were:

Age Type — Germany’s Culturally Elderly population is more like to become Arvato Financials Services prospective customers.
Buyer Type — Customers more than 2 years with the Arvato Financials Services are more likely to buy another product.
Customer journey typology- Individuals who are seeking orientation are more likely to become customers than their counterpart.
TECHNOLOGY — Here our analysis predicted that Individuals who have indifferent mindset with respect to Technology product group are likely to become Arvato Financials Services prospective customers.
Mail-Order Online/Mail-Order Online — Arvato Financials Services Customers with highest online/offline activity in last 3 years are likely to buy another financial product with the company.
BOOKS and CDS — BOOKS and CDS Buyers > 24 months of purchasing history are more likely to become Arvato Financials Services Customers.
HOUSE DECORATION — Individuals with who have made a purchase of HOUSE DECORATION Buyers > 24 months or higher are likely to become customers.
TOTAL — Individuals with 100% Online-transactions within the last 12 months are extremally likely to become Arvato Financials Services customers.
DIETARY SUPPLEMENTS — Our Analysis was able to predict that individuals who have a transactional activity in DIETARY SUPPLEMENTS for 24 months or higher are likely to become Arvato Financials Services customer as well.
ALL OTHER CATEGORIES — Finally our analysis also predicted that individuals whose transactional activity is extremally high in All Other product Categories are very likely to become customers of Arvato Financials Services as well.

Licensing,Authors and Acknoledgements

Data Source:   Udacity_AZDIAS_052018.csv,
               Udacity_CUSTOMERS_052018.csv,
               Udacity_MAILOUT_052018_TRAIN.csv,
               Udacity_MAILOUT_052018_TEST.csv,
               DIAS Information Levels - Attributes 2017.xlsx,
               DIAS Attributes - Values 2017.xlsx

Acknoledgements: Udacity and Bertelsmann Arvato Analytics
                  


Refferences I used to complete this project are:

https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

https://www.yourdatateacher.com/2021/06/07/precision-recall-accuracy-how-to-choose/#:~:text=We%20use%20precision%20when%20we,many%20real%201%20as%20possible.
