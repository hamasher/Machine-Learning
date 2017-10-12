#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

# creating a list of all the exercised stock options except for NaN values
exercised_stock_options = []
for i in data_dict.values():
	x = i["exercised_stock_options"]
	if not x == "NaN":
		exercised_stock_options.append(x)

#print exercised_stock_options max and min values
print "The max exercised stock is", max(exercised_stock_options)
print "The min exercised stock is", min(exercised_stock_options)

# creating a list of the all the salaries except for NaN values
salary = []
for i in data_dict.values():
	x = i["salary"]
	if not x == "NaN":
		salary.append(x)

#print salary max and min values
print "The max salary is", max(salary)
print "The min salary is", min(salary)

### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
#feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
# Takes a list of features ('features_list'), searches the data dictionary for those features, 
# and returns those features in the form of a data list.
data = featureFormat(data_dict, features_list )
# Splits the data list, created by the previous statement, into poi and features 
# (in this case because the list in the first step were all financial features, we name the features 'finance_features').
poi, finance_features = targetFeatureSplit( data )




# Rescaling Features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
finance_features = np.array(finance_features)
rescaled_finance_features = scaler.fit_transform(finance_features)

financial_features_test = np.array([200000, 1000000])
financial_features_test_transformed = scaler.transform(np.float32(financial_features_test))

print financial_features_test_transformed


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans 
clf = KMeans(n_clusters=2)
clf.fit( finance_features )
pred = clf.predict( finance_features )


### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
	print "no predictions object named pred found, no clusters to plot"



