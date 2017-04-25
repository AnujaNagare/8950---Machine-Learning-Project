### @author  Anuja Pradeep Nagare 
###          anuja.nagare@uga.edu
### @version 25/04/2017

import time
start_time = time.time()

import sklearn
import numpy as np  
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd

# read data
dataframe = pd.read_csv('new_dataset.csv')

# create input X_values and y_values
X_values = dataframe.ix[:,['age','job','marital','education','default', 'housing', 'loan','contact','month','day_of_week','duration','campaign','pdays','previous','poutcome','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m','nr_employed']]
y_values = dataframe['y']

# Split the data as training and testing data
X_train, X_test, y_train, y_test  = train_test_split(X_values, y_values, test_size=0.2, random_state=0)


my_df_prediction = pd.DataFrame(columns=['DT','KNN','SVM_L','SVM_RBF', 'RFC', 'ABC', 'QDAC' ,'LogReg'])


#1 Classification using Decision Trees 
from sklearn import tree 
from sklearn.model_selection import cross_val_score
clf = tree.DecisionTreeClassifier()  
clf = clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
print("1. Decision Trees ", prediction)
scores_clf = cross_val_score(clf, X_values, y_values, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores_clf.mean(), scores_clf.std() * 2))

sLength = len(prediction)
my_df_prediction['DT'] = np.random.randn(sLength)
my_df_prediction['DT'] = pd.DataFrame(prediction)


#2 Classification using K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn = knn.fit(X_train,y_train)
prediction = knn.predict(X_test)
print("2. K Nearest Neighbors ", prediction)
scores_knn = cross_val_score(knn, X_values, y_values, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores_knn.mean(), scores_knn.std() * 2))

my_df_prediction['KNN'] = pd.DataFrame(prediction)

#3 Classification using Linear SVM
from sklearn.svm import SVC
svc_l = SVC(kernel="linear", C=0.025)
svc_l = svc_l.fit(X_train,y_train)
prediction = svc_l.predict(X_test)
print("3. Linear SVM ", prediction)
scores_svcl = cross_val_score(svc_l, X_values, y_values, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores_svcl.mean(), scores_svcl.std() * 2))

my_df_prediction['SVM_L'] = pd.DataFrame(prediction)

#4 Classification using RBF SVM  
from sklearn.svm import SVC
svc_rbf = SVC(gamma=1, C=2)
svc_rbf = svc_rbf.fit(X_train,y_train)
prediction = svc_rbf.predict(X_test)
print("4. RBF SVM  ", prediction)
scores_svc_rbf = cross_val_score(svc_rbf, X_values, y_values, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores_svc_rbf.mean(), scores_svc_rbf.std() * 2))

my_df_prediction['SVM_RBF'] = pd.DataFrame(prediction)


#5 Classification using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
rfc = rfc.fit(X_train,y_train)
prediction = rfc.predict(X_test)
print("5. RandomForestClassifier", prediction)
scores_rfc = cross_val_score(rfc, X_values, y_values, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores_rfc.mean(), scores_rfc.std() * 2))

my_df_prediction['RFC'] = pd.DataFrame(prediction)

#6 Classification using AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier()
abc = abc.fit(X_train,y_train)
prediction = abc.predict(X_test)
print("6. AdaBoostClassifier", prediction)
scores_abc = cross_val_score(abc, X_values, y_values, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores_abc.mean(), scores_abc.std() * 2))

my_df_prediction['ABC'] = pd.DataFrame(prediction)


#7 Classification using Quadratic Discriminant Analysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qdac = QuadraticDiscriminantAnalysis()
qdac = qdac.fit(X_train,y_train)
prediction = qdac.predict(X_test)
print("7. Quadratic Discriminant Analysis", prediction)
scores_qdac = cross_val_score(qdac, X_values, y_values, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores_qdac.mean(), scores_qdac.std() * 2))

my_df_prediction['QDAC'] = pd.DataFrame(prediction)


#8 Classification using logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg = logreg.fit(X_train,y_train)
prediction = logreg.predict(X_test)
print("8. LogisticRegression", prediction)

scores_logreg = cross_val_score(logreg, X_values, y_values, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores_logreg.mean(), scores_logreg.std() * 2))

my_df_prediction['LogReg'] = pd.DataFrame(prediction)


score = [[' ' ,'Scores_mean', 'Scores_std'], 
	     ['Decision Tree',scores_clf.mean(), scores_clf.std()], 
		 ['knn', scores_knn.mean(), scores_knn.std()],
		 ['Linear SVM',scores_svcl.mean(), scores_svcl.std()],
		 ['RBF SVM', scores_svc_rbf.mean(), scores_svc_rbf.std()],
		 ['RandomForestClassifier', scores_rfc.mean(), scores_rfc.std()],
		 ['AdaBoostClassifier', scores_abc.mean(), scores_abc.std()],
		 ['QuadraticDiscriminantAnalysis', scores_qdac.mean(), scores_qdac.std()],
		 ['LogisticRegression', scores_logreg.mean(), scores_logreg.std()]
		 ]

my_df = pd.DataFrame(score)
my_df.to_csv('scores.csv', index=False, header=False)

my_df_prediction.to_csv('prediction.csv', index=False, header=True)

print("--- %s seconds ---" % (time.time() - start_time))

