
# coding: utf-8

# In[1]:

import sys
import pickle
sys.path.append("../../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import pandas
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from pprint import pprint

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

from tester import test_classifier


# In[2]:

features_list = ['poi','salary', 'bonus', 'total_stock_value', 'shared_receipt_with_poi', 'bonus_to_salary_ratio',
                 'exercised_stock_options', 'poi_email', 'poi_to_total_emails']


# In[3]:

#loading the enron data set as data_dict dictionary
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[4]:

# starting the exploratory data analysis with some basic information about the dataset
print "There are", len(data_dict), "people in the Enron dataset."


# In[5]:

def number_of_pois(dict):
    count = 0 
    for employee in dict:
        if dict[employee]['poi'] == True:
            count += 1
    print "There are " + str(count) + " pois in the Enron dataset."

number_of_pois(data_dict)


# In[6]:

count = 0
with open('poi_names.txt', 'r') as f:
    for line in f:
        if '(y)' in line:
            count += 1
        if '(n)' in line:
            count+=1
print "There were", count, "pois total."


# In[7]:

# one of the best ways to find outliers in the dataset is visualization
features = ["bonus", "salary"]
data = featureFormat(data_dict, features)

print data.max()
for point in data:
    bonus = point[0]
    salary = point[1]
    plt.scatter( bonus, salary )

plt.xlabel("bonus")
plt.ylabel("salary")
plt.show()


# In[8]:

bonus_outliers = []
for key in data_dict:
    value = data_dict[key]['bonus']
    if value == 'NaN':
        continue
    bonus_outliers.append((key,int(value)))

pprint(sorted(bonus_outliers,key=lambda x:x[1],reverse=True)[:2])


salary_outliers = []
for key in data_dict:
    value = data_dict[key]['salary']
    if value == 'NaN':
        continue
    salary_outliers.append((key,int(value)))

pprint(sorted(salary_outliers,key=lambda x:x[1],reverse=True)[:2])


# Based on this analysis, we can see that the "TOTAL" value was accidentally added to the dataset. Upon further investigation of the PDF file, we can see that "THE TRAVEL AGENCY IN THE PARK" was also mistakenly added to the dataset. Addtionally, "LOCKHART, EUGENE E" has no values for any of the variables. Before removing these values, however, I want to see how they appear in the dictionary.

# In[9]:

for key, value in data_dict.iteritems():
    print key


# In[10]:

#Removing the outliers mentioned above
outliers = ['TOTAL','THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']

for outlier in outliers:
    data_dict.pop(outlier, 'none')


# In[11]:

for key, value in data_dict.iteritems():
    print key

#checking the length of the dictionary to make sure that the three values were removed correctly
print len(data_dict)


# In[12]:

#want to create a dataframe from the dictionary to perform exploratory data analysis
df = pandas.DataFrame.from_records(list(data_dict.values()))
employees = pandas.Series(list(data_dict.keys()))


# In[13]:

print df


# In[14]:

#checking some basic descriptive statistics for the dataframe
print df.describe()


# In[15]:

# want to replace NaN values with a numerical NaN for calculations
df.replace('NaN', np.nan, inplace = True)


# In[16]:

# want to see the correlation between poi and the different columns
df.corr(method='pearson')


# Surpisingly, the only feature with a moderate correlation, one with a pearson's correlation value greater than +/-0.5, was loan_advances (0.560178).

# In[17]:

# going to drop the "director_fee", "email_address", "restricted_stock_deferred", and "restricted_stock" columns since there 
# is either no, or negligable, correlation with poi
df.drop(['director_fees', 
         'email_address',
         'restricted_stock_deferred',
         'restricted_stock'],
        axis=1,inplace=True)


# In[18]:

#checking the integer types annd also making sure the columns above were removed
print df.dtypes


# In[19]:

# want to group the data points by poi and find the mean values for each of the features
for ind in df:
    if ind != 'poi':
        print '\n', ind
        df_temp = df[['poi',ind]].dropna().copy()
        print df_temp.groupby('poi').mean()


# Looking at the mean values for the differenet metrics, grouped by POI. Total_stock_value, loan_advances, total_payments, and exercised_stock_options appear to have the biggest differences between the mean values for POIs and non-POIs. 

# In[20]:

#want to visualize the data by creating boxplots of the different features 
   
def boxplots_by_poi():
    for ind in df:
        if ind != 'poi':
            df_temp = df[['poi', ind]].dropna().copy()
            quantile = df[ind].quantile(.9)
            ax = df_temp.boxplot(by = 'poi')
            ax.set_ylim(0, quantile)
            ax.set_title(' ')
            ax.set_ylabel(ind)
            plt.suptitle(' ')
            plt.show()

print boxplots_by_poi()


# The boxplots are created to give a visual representation of the difference in values between the POIs and non-POIs. It is important to visualize the results because it provides us a better understanding of what some of the previously employed metrics really mean. For example, the highest correlation with POI detection was "loan_advances", but the graph shows that there is not much of a difference between loan_advances of POIs and non-POIs. Although POIs and non-POIs have differences in most metrics, there are significant differences for the "bonus", "exercised_stock_options", "from_poi_to_this_person", "to_poi_from_this_person", "long_term_incentive", "shared_receipt_with_poi" and "total_stock_value" features.

# In[21]:

#created a new variable called bonus_to_salary ratio
df['bonus_to_salary_ratio'] = df.bonus.div(df.salary, axis=0)

print df['bonus_to_salary_ratio'], df['bonus'], df['salary']
print df['bonus_to_salary_ratio'].corr(df["poi"], method='pearson')


# In[22]:

df['poi_email'] = df.from_this_person_to_poi + df.from_poi_to_this_person + df.shared_receipt_with_poi

print df['poi_email'], df['from_this_person_to_poi'], df["from_poi_to_this_person"], df['shared_receipt_with_poi']
print df['poi_email'].corr(df["poi"], method='pearson')


# In[23]:

#creating a total_messages variable
df['total_messages'] = df['from_messages'] + df['to_messages']

#creating a "poi_to_total_emails" variable which is the ratio of poi emails to total_messages
df['poi_to_total_emails'] = df['poi_email']/ df['total_messages']

df['poi_to_total_emails'].corr(df["poi"], method='pearson')


# Want to create a new variable, "now_versus_later", that is the ratio between money that someone is set to receive now versus a later time. Poi's have a greater incentive to receive the money immediately because they know the company's future is bleak. The variables used to create this new variable are expenses, deferral payment, loan advances, long term incentive, and deferred income. Expenses are fees from consulting services and reimbursements from the company and deferral payments are distributions from deferred compensation, so they must be paid now. Loan advances, which are loans in return of a promise of repayment, long term incentives, which typically last 3-5 years, and deferred income are all items that must be paid at later time.

# In[24]:

df['now'] = df['expenses'] + df['deferral_payments']

print df['now']

df['later'] = df['loan_advances'] + df['long_term_incentive'] + df['deferred_income']

print df['later']

df['now_versus_later'] = (df['expenses'] + df['deferral_payments'])/(1 + df['loan_advances'] + df['long_term_incentive'] + df['deferred_income'])

print df['now_versus_later']


# In[25]:

print df['now'].corr(df["poi"], method='pearson')
print df['later'].corr(df["poi"], method='pearson')
print df['now_versus_later'].corr(df["poi"], method='pearson')


# In[26]:

df['fraction_of_deferred_income_to_total_payments'] = df['deferred_income']/df['total_payments']

print df['fraction_of_deferred_income_to_total_payments']
print df['fraction_of_deferred_income_to_total_payments'].corr(df["poi"], method='pearson')


# In[27]:

list(df)


# In[28]:

df.replace(np.nan, 0, inplace = True)


# In[29]:

# setting the index of df to be the employees series
df.set_index(employees, inplace=True)


# In[30]:

# creating a list of the column names
new_features_list = df.columns.values


# In[31]:

# creating a dictionary from the dataframe
df_dict = df.to_dict('index')


# In[32]:

# comparing the original dictionary with the dictionary reconstructed from the df
print df_dict == data_dict


# In[33]:

# Extracting features and labels from dataset for local testing
data = featureFormat(df_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


# In[34]:

# Naive-Bayes Classifier
clf = GaussianNB()

test_classifier(clf, df_dict, features_list)


# In[35]:

#Decision Tree Classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=2)

test_classifier(clf, df_dict, features_list)


# In[36]:

#scaler = MinMaxScaler()
#test_data = np.array(features_test)
#scaled_test_data = scaler.fit_transform(test_data)

#KMeans
#clf = KMeans(n_clusters=2)

#test_classifier(clf, df_dict, features_list)


# In[37]:

# Random Forests
#clf = RandomForestClassifier(n_estimators=9)

#test_classifier(clf, df_dict, features_list)


# In[38]:

#from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier()

#test_classifier(clf, df_dict, features_list)


# In[39]:

#clf =  KNeighborsClassifier(n_neighbors = 3, weights = 'distance')

#test_classifier(clf, df_dict, features_list)


# In[40]:

# Linear SVM
#from sklearn.svm import SVC
#clf = SVC(kernel="linear", C=0.025)

#from sklearn.neighbors import KNeighborsClassifier


# In[41]:

# LINEAR SVM
#from sklearn.svm import LinearSVC
#clf = SVC(kernel= 'rbf', gamma=2, C=1)

#from sklearn.neighbors import KNeighborsClassifier


# Now I will do feature selection.

# In[42]:

#selecting specific features
np.random.seed(42)

#selectKBest
skb = SelectKBest(f_classif, k = 'all')
skb.fit(features_train, labels_train)

# Get Features Selected
features_selected=[features_list[i+1] for i in skb.get_support(indices=True)]
print 'The Features Selected by SKB:'
print features_selected

# 2. SelectKBest - GridSearchCV
skb = SelectKBest(f_classif)

pipeline =  Pipeline(steps=[("SKB", skb), ("NaiveBayes", GaussianNB())])
params_skb = {'SKB__k': range(1,8)}
          
gs = GridSearchCV(pipeline, param_grid = params_skb, scoring = 'f1')
gs.fit(features_train, labels_train)

# Get Features Selected
features_selected=[features_list[i+1] for i in gs.best_estimator_.named_steps['SKB'].get_support(indices=True)]
print 'The Features Selected by SKB - GS:'
print features_selected


# In[43]:

k_best = gs.best_estimator_.named_steps['SKB']

# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in k_best.scores_ ]
# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  k_best.pvalues_ ]
# Get SelectKBest feature names, whose indices are stored in 'K_best.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in k_best.get_support(indices=True)]

# Sort the tuple by score, in reverse order
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

print features_selected_tuple


# In[44]:

#want to select a smaller number of folds to make sure that it runs quickly
folds = 10
kbest = SelectKBest(f_classif)
 
#use StratifiedShuffleSplit to ensure that the class sizes are the same
sss = StratifiedShuffleSplit(labels, 100, test_size = 0.3, random_state = 42)
 
#test the default KNeighbors Classifier
NB = GaussianNB()
 
#use Pipeline to chain SelectKBest and KNearestNeighbors Classifer
pipeline = Pipeline([("scaler", MinMaxScaler()), ('kbest', kbest), ('NaiveBayes', NB)])
 
# need to create a parameter grid
param_grid = { 'kbest__k': range(1,5),
             }
 
gs = GridSearchCV(pipeline, param_grid, scoring = 'f1',cv=sss)
gs.fit(features, labels)
clf_nb = gs.best_estimator_
bp = gs.best_params_
clf_nb = gs.best_estimator_
 
from tester import test_classifier
test_classifier(clf, df_dict, features_list)


# In[45]:
# should not use unsupervised algorithm for this dataset
#from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import StratifiedShuffleSplit
#want the number of folds to be balanced between speed and randomness
#folds = 10
#kbest = SelectKBest(f_classif)
 
#use StratifiedShuffleSplit to ensure that the class sizes are the same
#sss = StratifiedShuffleSplit(labels, folds, test_size = 0.3, random_state = 42)
 
#test the default KNeighbors Classifier
#knn =  KNeighborsClassifier()
 
#use Pipeline to chain SelectKBest and KNearestNeighbors Classifer
#pipeline = Pipeline([('kbest', kbest), ('KNeighbors', knn)])
 
# need to create a parameter grid
#param_grid = { 'kbest__k': range(1,8),
              #'KNeighbors__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              #'KNeighbors__leaf_size': [2,10,20,30],
              #'KNeighbors__metric': ['minkowski', 'manhattan', 'euclidean'],
              #'KNeighbors__n_neighbors': [1, 2, 3, 4, 5, 6, 7],
              #'KNeighbors__weights': ['uniform', 'distance']
             #}
 
#gs = GridSearchCV(pipeline, param_grid, scoring = 'f1',cv=sss)
#gs.fit(features, labels)
#clf = gs.best_estimator_
#bp = gs.best_params_
#clf = gs.best_estimator_
 
#from tester import test_classifier
#test_classifier(clf, df_dict, features_list)


# In[46]:

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import tree

#want the same number of folds as tester.py
folds = 1000
kbest = SelectKBest(f_classif)
 
#use StratifiedShuffleSplit to ensure that the class sizes are the same
sss = StratifiedShuffleSplit(labels, 100, test_size = 0.3, random_state = 42)
 
#test the default KNeighbors Classifier
clf =  tree.DecisionTreeClassifier()
 
#use Pipeline to chain SelectKBest and KNearestNeighbors Classifer
pipeline = Pipeline([('kbest', kbest), ('DecisionTree', clf)])
 
# need to create a parameter grid
param_grid = { 'kbest__k': range(1,5),
             }
 
gs = GridSearchCV(pipeline, param_grid, scoring = 'f1',cv=sss)
gs.fit(features, labels)
clf = gs.best_estimator_
bp = gs.best_params_
clf = gs.best_estimator_
 
from tester import test_classifier
test_classifier(clf, df_dict, features_list)

dump_classifier_and_data(clf, df_dict, features_list)