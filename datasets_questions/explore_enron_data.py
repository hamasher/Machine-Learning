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

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#print "There are", len (enron_data), "people in the dataset"
    
# prints each key with the number of features that key has
#for key, value in enron_data.items():
    #print(key, len([item for item in value if item]))


#pois = 0
#for v in enron_data.values():
 #   if v["poi"]: 
  #  	pois += 1
#print "There are", pois, "persons of interest"

#person of interest via the text file
#count = 0
#with open('poi_names.txt', 'r') as f:
 #   for line in f:
  #      if '(y)' in line:
   #         count += 1
    #    if '(n)' in line: 
     #   	count+=1

#print "there are", count, "persons on interest in the text file"

#James Prentice total stock value
#Prentice_stock = enron_data['PRENTICE JAMES']["total_stock_value"]
#Colwell_poi = enron_data['COLWELL WESLEY']['from_this_person_to_poi']
#Skilling_exercised_stock = enron_data['SKILLING JEFFREY K']['exercised_stock_options']
#Skilling_payments = enron_data['SKILLING JEFFREY K']['total_payments']
#Lay_payments = enron_data['LAY KENNETH L']['total_payments']
#Fastow_Payments = enron_data['FASTOW ANDREW S']['total_payments']


#print Prentice_stock
#print Colwell_poi
#print Skilling_exercised_stock
#print "Skilling got paid $", Skilling_payments 
#print "Lay got paid $", Lay_payments
#print "Fastow got paid $", Fastow_Payments

#salary_distinct = 0
#email_distinct = 0
#for v in enron_data.values():
#	if v["salary"] != 'NaN': 
#		salary_distinct += 1
#	if v["email_address"] != 'NaN':
#		email_distinct +=1

#print "the number of people with distinct salaries is", salary_distinct
#print "the number of people with disticnt email addresses", email_distinct

#payments = sum([item["total_payments"]=='NaN' for item in enron_data.values()])
#percent = (float(payments)/len(enron_data)) * 100
#print payments, percent

pois = 0
count = 0
for v in enron_data.values():
    if v["poi"]:
        pois += 1
        if v["total_payments"] == 'NaN': 
        	count += 1

print pois
print count