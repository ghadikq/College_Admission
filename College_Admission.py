import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import math
from statsmodels.formula.api import ols

coladm = pd.read_csv('C:\\Users\\ghadi\\Documents\\DataSet\\College_admission.csv')

# preform EDA 
coladm.head()
coladm.dtypes
coladm.info()
coladm.shape
coladm.columns

# drop na values if there is any
coladm.dropna()

# change columns types so its suitable 
#df = df.astype(dtype) this used to change all df to same type
coladm['admit'] = coladm['admit'].astype('category')
coladm['ses'] = coladm['ses'].astype('category')
coladm['Gender_Male'] = coladm['Gender_Male'].astype('category')
coladm['Race'] = coladm['Race'].astype('category')
coladm['rank'] = coladm['rank'].astype('category')

# data correlation to select featuers 
coladm.drop(columns=['admit']).corrwith(coladm['admit'])
#coladm[['gpa','gre','admit']].corr()['admit'][:] to see corr with certain featuers only

# Data Visualization

## visualize the numeric values to see data & have insight on data
sns.displot(coladm, x="gpa", kind="kde",color="Green")
sns.displot(coladm, x="gre", kind="kde",color="Green")

# see admit result based on gpa
sns.barplot(x="admit", y="gpa",data=coladm,palette="Greens")
# see admit result based on gre
sns.barplot(x="admit", y="gre",data=coladm,palette="Greens")

# check for outliers
sns.boxplot(x="admit", y="gpa", data=coladm,palette="Greens")
sns.boxplot(x="admit",y="gre",data=coladm,palette="Greens")

# as you can see in the plot there are 3 point[outliers] which is most likely they are the outliers in this data

# remove outliers by z-score
coladm.shape # shape before remove outliers
z_scores = stats.zscore(coladm) # find z-score
abs_z_scores = np.abs(z_scores) #detect the outliers
coladm = coladm[(abs_z_scores < 3).all(axis=1)]# remove outliers & assign cleaned dataset to coladm
coladm_c.shape # shape after remove outliers

# check again for outliers by seeing boxplot
sns.boxplot(x="admit", y="gpa", data=coladm_c,palette="Greens")
sns.boxplot(x="admit",y="gre",data=coladm_c,palette="Greens")


## shapiro test normality test
#to compute how likely it is for a random variable underlying the data set to be normally distributed
### for gpa:
shapiro_test_gpa = stats.shapiro(coladm['gpa'])
shapiro_test_gpa
### for gre:
shapiro_test_gre = stats.shapiro(coladm['gre'])
shapiro_test_gre

## applying log transformation and retest 
## the values using Shapiro Test the results
## to make it close to normal distribution as much as possible
## BUT, after doing this in R I did not see much different so I don't think this is nessecary but I wanted to try doing shapiro test in Python


# split data to train & test
# pip install "sklearn" # install sklearn
import sklearn
from sklearn.model_selection import train_test_split
y = coladm.admit
X = coladm.drop('admit',axis=1)
X_train, X_test, y_train, y_test = train_test_split(coladm.drop('admit',axis=1), coladm['admit'], test_size=0.30)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
# See Size for each split
X_train.shape
X_test.shape
y_train.shape
y_test.shape

# Model Import
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

# Logistic Regression

logmodel = LogisticRegression()# create model
logmodel.fit(X_train,y_train)# fit model
logpred = logmodel.predict(X_test)# fit Logistic Regression model on test data set

# check the result by having classification report for model
print(classification_report(y_test,logpred)) # Accuracy is 66%
# also use confusion matrix to see prediction results
logconfusion_matrix = confusion_matrix(y_test, logpred)
print(logconfusion_matrix) # this is not really good model

# cross-validation procedure
logcv = KFold(n_splits=10, random_state=1, shuffle=True)
logscores = cross_val_score(logmodel, X_train, y_train, scoring='accuracy', cv=logcv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(logscores), std(logscores))) #Accuracy is 70.8% ,after validation



# GBM

GBMmodel = GradientBoostingClassifier()# create model
GBMmodel.fit(X_train,y_train)# fit model
GBMpred = GBMmodel.predict(X_test)# predict test values(outcome)
print(classification_report(y_test,GBMpred)) #Accuracy is 66% , classification report to check model results                          
GBMconfusion_matrix = confusion_matrix(y_test, GBMpred)# to see prediction results using confusion matrix
print(GBMconfusion_matrix)# the model has small improve 

# cross-validation procedure
gcv = KFold(n_splits=10, random_state=1, shuffle=True)
gscores = cross_val_score(GBMmodel, X_train, y_train, scoring='accuracy', cv=gcv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(gscores), std(gscores))) #Accuracy is 66.8% ,after validation



# Random Forest Clssifier

rf_model = RandomForestClassifier() # create model
rf_model.fit(X_train,y_train) # fit model
rf_pred = rf_model.predict(X_test) # predict test values(outcome)
print(classification_report(y_test,rf_pred)) # Accuracy 62% , classification report to check model results
rf_confusion_matrix = confusion_matrix(y_test, rf_pred)# to see prediction results using confusion matrix
print(rf_confusion_matrix)

# cross-validation procedure
rfcv = KFold(n_splits=10, random_state=1, shuffle=True) # regular Kfolds 
rfscores = cross_val_score(rf_model, X_train, y_train, scoring='accuracy', cv=rfcv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores))) #Accuracy is 66.8% ,after validation


