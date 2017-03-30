#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:26:16 2017

@author: rasam
"""
#Import necessary libraries
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Load the data.
online_news = pd.read_csv('/Users/rasam/Google Drive/Spring17/PSWD/PSW_mid_term/OnlineNewsPopularity/OnlineNewsPopularity.csv', sep =',')
#----------------------------ANSWER 1-----------------------------------------------------------------
#Calculate correlation
correlations = {}
columns = online_news.columns.tolist()

for col_a, col_b in itertools.combinations(columns, 2):
    correlations[col_a + '__' + 'shares'] = np.corrcoef(online_news.loc[:, col_a], online_news.loc[:, 'shares'])

#Train the model.
y = online_news.shares
X = online_news.abs_title_sentiment_polarity
X = sm.add_constant(X)

#Plot the scatter plot.
lr_model = sm.OLS(y,X).fit()
print(lr_model.summary())


X_prime = np.linspace(online_news.abs_title_sentiment_polarity.min(), online_news.abs_title_sentiment_polarity.max(), 100)
X_prime = sm.add_constant(X_prime)
y_hat = lr_model.predict(X_prime)

plt.figure()
plt.scatter(X.abs_title_sentiment_polarity, y)
plt.xlabel("abs_title_sentiment_polarity")
plt.ylabel("Shares")
plt.plot(X_prime[:,1], y_hat, 'red')


#----------------------------ANSWER 2--------------------------------------------------------------
#Making subset of news those were published on weekends.

weekend_news = online_news[online_news['is_weekend']==1]

#Calculate correlation
correlations = {}
columns = weekend_news.columns.tolist()

for col_a, col_b in itertools.combinations(columns, 2):
    correlations[col_a + '__' + 'shares'] = np.corrcoef(weekend_news.loc[:, col_a], weekend_news.loc[:, 'shares'])

plt.figure()
plt.subplot(211)
plt.xlabel('Shares')
plt.ylabel('abs_title_sentiment_polarity')
plt.scatter(weekend_news.shares, weekend_news.abs_title_sentiment_polarity)

plt.subplot(212)
plt.xlabel('Shares')
plt.ylabel('title_sentiment_polarity')
plt.scatter(weekend_news.shares, weekend_news.title_sentiment_polarity)

#Developing regression model.

import statsmodels.api as sm

y = weekend_news.shares
X = weekend_news[['abs_title_sentiment_polarity', 'title_sentiment_polarity']]
X = sm.add_constant(X)

lr_model = sm.OLS(y,X).fit()
print(lr_model.summary())

from mpl_toolkits.mplot3d import Axes3D

X_Axis, Y_Axis = np.meshgrid(np.linspace(X.abs_title_sentiment_polarity.min(), X.abs_title_sentiment_polarity.max(),100),
                             np.linspace(X.title_sentiment_polarity.min(), X.title_sentiment_polarity.max(),100))

Z_Axis = lr_model.params[0] + lr_model.params[1]*X_Axis + lr_model.params[2]*Y_Axis

fig = plt.figure(figsize=(7,7))

ax = Axes3D(fig, azim = 120)

ax.plot_surface(X_Axis, Y_Axis, Z_Axis, alpha =0.5, cmap = plt.cm.coolwarm)

ax.scatter(X.abs_title_sentiment_polarity, X.title_sentiment_polarity, y)

ax.set_xlabel('abs_title_sentiment_polarity')
ax.set_ylabel('title_sentiment_polarity')
ax.set_zlabel('Shares')

#----------------------------ANSWER 3--------------------------------------------------------------
#Making subset of news those were published on weekdays.

weekday_news = online_news[online_news['is_weekend']==0]

#Calculate correlation
correlations = {}
columns = weekday_news.columns.tolist()

for col_a, col_b in itertools.combinations(columns, 2):
    correlations[col_a + '__' + 'shares'] = np.corrcoef(weekday_news.loc[:, col_a], weekday_news.loc[:, 'shares'])

#Train the model.
plt.figure()
plt.subplot(211)
plt.xlabel('Shares')
plt.ylabel('abs_title_sentiment_polarity')
plt.scatter(weekday_news.shares, weekday_news.abs_title_sentiment_polarity)

plt.subplot(212)
plt.xlabel('Shares')
plt.ylabel('title_subjectivity')
plt.scatter(weekday_news.shares, weekday_news.title_subjectivity)

#Developing regression model.

import statsmodels.api as sm

y = weekday_news.shares
X = weekday_news[['abs_title_sentiment_polarity', 'title_subjectivity']]
X = sm.add_constant(X)

lr_model = sm.OLS(y,X).fit()
print(lr_model.summary())

from mpl_toolkits.mplot3d import Axes3D

X_Axis, Y_Axis = np.meshgrid(np.linspace(X.abs_title_sentiment_polarity.min(), X.abs_title_sentiment_polarity.max(),100),
                             np.linspace(X.title_subjectivity.min(), X.title_subjectivity.max(),100))

Z_Axis = lr_model.params[0] + lr_model.params[1]*X_Axis + lr_model.params[2]*Y_Axis

fig = plt.figure(figsize=(7,7))

ax = Axes3D(fig, azim = 120)

ax.plot_surface(X_Axis, Y_Axis, Z_Axis, alpha =0.5, cmap = plt.cm.coolwarm)

ax.scatter(X.abs_title_sentiment_polarity, X.title_subjectivity, y)

ax.set_xlabel('abs_title_sentiment_polarity')
ax.set_ylabel('title_subjectivity')
ax.set_zlabel('Shares')

#----------------------------ANSWER 5--------------------------------------------------------------

newsdata_low = online_news[online_news.shares<1400]
newsdata_high = online_news[online_news.shares>=1400]

plt.plot(newsdata_low['abs_title_sentiment_polarity'],newsdata_low['title_subjectivity'],'o')
plt.plot(newsdata_high['abs_title_sentiment_polarity'],newsdata_high['title_subjectivity'],'o')
plt.xlabel('abs_title_sentiment_polarity')
plt.ylabel('title_subjectivity')
plt.show()

#----------------------------ANSWER 6--------------------------------------------------------------
#Clustering

Xl = np.array(online_news['title_subjectivity'])
Yl =  np.array(online_news['abs_title_sentiment_polarity'])


features_l = np.concatenate(([Xl],[Yl]),axis=0)
features_l = np.transpose(features_l)


plt.scatter(features_l[:,0], features_l[:,1], marker = "o")

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(features_l)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

colors = ["g.","r.","b."]
colors_given=[]
for i in range(len(features_l)):
    plt.plot(features_l[i][0], features_l[i][1], colors[labels[i]])
plt.title('Clustering for K=2')
plt.xlabel('title_subjectivity')
plt.ylabel('abs_title_sentiment_polarity')
plt.scatter(centroids[:,0],centroids[:,1], marker="x")


#K=3

kmeans = KMeans(n_clusters=3)
kmeans.fit(features_l)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

colors = ["g.","r.","b."]
colors_given=[]
for i in range(len(features_l)):
    plt.plot(features_l[i][0], features_l[i][1], colors[labels[i]])
plt.title('Clustering for K=3')
plt.xlabel('title_subjectivity')
plt.ylabel('abs_title_sentiment_polarity')
plt.legend('low','high','unkown')
plt.scatter(centroids[:,0],centroids[:,1], marker="x")

#----------------------------ANSWER 7--------------------------------------------------------------
#Create categorical attribute 'pop' which is 1 if shares >=1400 else 0
online_news['pop'] = np.where(online_news['shares']>=1400, 1, 0)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(online_news[['title_subjectivity','abs_title_sentiment_polarity']], online_news['pop'], test_size=0.3)
from sklearn.neighbors import KNeighborsClassifier
results = []
for k in range(3,9, 2):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train,y_train)
            
    prediction = classifier.predict(X_test)
            
    correct = np.where(prediction== y_test,1, 0).sum()
    accuracy = correct/len(y_test)
    results.append([k,accuracy])

results= pd.DataFrame(results, columns = ["k", "accuracy"])

plt.xlabel('K')
plt.ylabel('Accuracy')
plt.plot(results.k, results.accuracy)
plt.show()

