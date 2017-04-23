import time
start_time = time.time()

import pandas as pd
# read data
dataframe = pd.read_csv('banking.csv')

# Data cleaning converting strings to numbers 
job = list(set(dataframe.ix[:,1]))
for i in range(0, len(dataframe.ix[:,1])):
	dataframe.ix[i,1] = job.index(dataframe.ix[i,1])

marital = list(set(dataframe.ix[:,2]))
for i in range(0, len(dataframe.ix[:,2])):
	dataframe.ix[i,2] = marital.index(dataframe.ix[i,2])

education = list(set(dataframe.ix[:,3]))
for i in range(0, len(dataframe.ix[:,3])):
	dataframe.ix[i,3] = education.index(dataframe.ix[i,3])

default = list(set(dataframe.ix[:,4]))
for i in range(0, len(dataframe.ix[:,4])):
	dataframe.ix[i,4] = default.index(dataframe.ix[i,4])

housing = list(set(dataframe.ix[:,5]))
for i in range(0, len(dataframe.ix[:,5])):
	dataframe.ix[i,5] = housing.index(dataframe.ix[i,5])

loan = list(set(dataframe.ix[:,6]))
for i in range(0, len(dataframe.ix[:,6])):
	dataframe.ix[i,6] = loan.index(dataframe.ix[i,6])

contact = list(set(dataframe.ix[:,7]))
for i in range(0, len(dataframe.ix[:,7])):
	dataframe.ix[i,7] = contact.index(dataframe.ix[i,7])

month = list(set(dataframe.ix[:,8]))
for i in range(0, len(dataframe.ix[:,8])):
	dataframe.ix[i,8] = month.index(dataframe.ix[i,8])

day_of_week = list(set(dataframe.ix[:,9]))
for i in range(0, len(dataframe.ix[:,9])):
	dataframe.ix[i,9] = day_of_week.index(dataframe.ix[i,9])

poutcome = list(set(dataframe.ix[:,14]))
for i in range(0, len(dataframe.ix[:,14])):
	dataframe.ix[i,14] = poutcome.index(dataframe.ix[i,14])

dataframe.to_csv('output.csv')

print("--- %s seconds ---" % (time.time() - start_time))
