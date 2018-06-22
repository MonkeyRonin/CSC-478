#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 23:12:48 2017

@author: Yiyang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Create Dataframe
pm_data = pd.read_csv("/Users/Yiyang/Documents/CSC 478/FInal Project/Pokemon.csv")
pm_data.head()

pm_df = pm_data.drop('#', axis = 1)
pm_df = pm_df.drop('Type 2', axis = 1)
pm_df.head()

#
pm_df.describe(include = "all")

#Dist Plot
hp = pm_df.ix[:, 3]
hp.head()
sns.distplot(hp)

atk = pm_df.ix[:, 4]
sns.distplot(atk)

defs = pm_df.ix[:, 5]
sns.distplot(defs)

spatk = pm_df.ix[:, 6]
sns.distplot(spatk)

spdef = pm_df.ix[:, 7]
sns.distplot(spdef)

spd = pm_df.ix[:, 8]
sns.distplot(spd)

#Boxplot
stat = []
for i in list(pm_df.columns.values)[3: 9]:
    stat.append(i)
stat

plt.figure(figsize = (10, 5))
sns.boxplot(pm_df[stat])

#Correlation
cor_matrix = pm_df[stat].corr()
plt.subplots(figsize = (7, 7))
sns.heatmap(cor_matrix, annot = True, square = True, annot_kws = {"size": 16}, cmap = 'Blues')

#PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pm_scale = StandardScaler().fit(pm_df[stat])
df_scale = pm_scale.transform(pm_df[stat])
print(df_scale[:, 0].mean())
print(df_scale[:, 0].std())
pm_pca = PCA(n_components = 0.8).fit(df_scale)

scores = pd.DataFrame(pm_pca.transform(df_scale))

loadings = pd.DataFrame(pm_pca.components_, columns = stat)

loading2 = loadings ** 2
pm_pca_heatmap = sns.heatmap(loading2.transpose(), linewidths = 0.5, cmap = "Blues", annot = True)
pm_pca_heatmap.set_xticklabels(pm_pca_heatmap.xaxis.get_majorticklabels(), rotation = 0, fontsize = 8)
pm_pca_heatmap.set_yticklabels(pm_pca_heatmap.yaxis.get_majorticklabels(), rotation = 0, fontsize = 8)

pm_pca_heatmap2 = sns.heatmap(loadings.transpose(), center = 0, linewidths = 0.5, cmap = "Blues", vmin = -1, vmax = 1, annot = True)
pm_pca_heatmap2.set_xticklabels(pm_pca_heatmap2.xaxis.get_majorticklabels(), rotation = 0, fontsize = 8)
pm_pca_heatmap2.set_yticklabels(pm_pca_heatmap2.yaxis.get_majorticklabels(), rotation = 0, fontsize = 8)

#KMeans Clustering
pm_pca_k = PCA(n_components = 0.8).fit_transform(df_scale)
from sklearn.cluster import KMeans
kmeans_pca = KMeans(n_clusters = 4).fit(pm_pca_k)
stat_cluster = pd.DataFrame(kmeans_pca.cluster_centers_.T, index = ['PC 1', 'PC 2', 'PC 3', 'PC 4'], columns = ['Cluster 1', 'Cluster 2','Cluster 3','Cluster 4'])
stat_cluster

plt.figure(figsize=(12, 5))
cmap = plt.get_cmap('nipy_spectral')
plt.subplot(1,2,2)
plt.scatter(pm_pca_k[:, 0], pm_pca_k[:, 1], c = cmap(kmeans_pca.labels_ / 4))
plt.title('PCA');


from sklearn.metrics import completeness_score, homogeneity_score
c_score = completeness_score(pm_df['Total'], kmeans_pca.labels_)
print(c_score)
h_score = homogeneity_score(pm_df['Total'], kmeans_pca.labels_)
print(h_score)

#Classification
pm_stat = pm_df[stat]
pm_stat.head()
pm_target = pm_df.Legendary
pm_target.head()

from sklearn.cross_validation import train_test_split
pm_train, pm_test, pm_target_train, pm_target_test = train_test_split(pm_stat, pm_target, test_size = 0.2, random_state = 33)

pm_train.shape
pm_test.shape
pm_target_train.shape
pm_target_test.shape

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler().fit(pm_train)
pm_train_norm = min_max_scaler.transform(pm_train)
pm_test_norm = min_max_scaler.transform(pm_test)

from sklearn import neighbors, tree, naive_bayes
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#KNN
nn = 5
knnclf = neighbors.KNeighborsClassifier(nn, weights = 'distance')
knnclf.fit(pm_train_norm, pm_target_train)
knnpreds_test = knnclf.predict(pm_test_norm)
print(knnpreds_test)
print(classification_report(pm_target_test, knnpreds_test))
knncm = confusion_matrix(pm_target_test, knnpreds_test)
print(knncm)

print(knnclf.score(pm_test_norm, pm_target_test))
print(knnclf.score(pm_train_norm, pm_target_train))

#Decision Tree
treeclf = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 3)
treeclf = treeclf.fit(pm_train, pm_target_train)
treepreds_test = treeclf.predict(pm_test)
print(treepreds_test)
print(treeclf.score(pm_test, pm_target_test))
print(treeclf.score(pm_train, pm_target_train))
print(classification_report(pm_target_test, treepreds_test))
treecm = confusion_matrix(pm_target_test, treepreds_test, labels = ['Yes','No'])
print(treecm)

#Naive Bayes
nbclf = naive_bayes.GaussianNB()
nbclf = nbclf.fit(pm_train, pm_target_train)
nbpreds_test = nbclf.predict(pm_test)
print(nbpreds_test)

print(nbclf.score(pm_train, pm_target_train))
print(nbclf.score(pm_test, pm_target_test))