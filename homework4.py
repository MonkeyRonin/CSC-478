#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 16:18:37 2017

@author: Yiyang
"""

import numpy as np
import pandas as pd

#1a.
s_data = pd.read_csv("/Users/Yiyang/Documents/CSC 478/segmentation_data/segmentation_data.txt", header = None)
s_data.head()

s_class = pd.read_csv("/Users/Yiyang/Documents/CSC 478/segmentation_data/segmentation_classes.txt", header = None, sep = '\t', names = ['Name','Value'])
s_class.head()

x_train = np.array(s_data)

s_names = pd.read_csv("/Users/Yiyang/Documents/CSC 478/segmentation_data/segmentation_names.txt", header = None)
s_names

s_name = s_names.ix[:, 0]
s_name

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler().fit(s_data)
s_data_norm = min_max_scaler.transform(s_data)
s_data_norm

#1b.
from sklearn.cluster import KMeans

kMeans = KMeans(n_clusters = 7)
kMeans.fit(s_data_norm)
kMeans.labels_
kMeans.cluster_centers_
pd.DataFrame(kMeans.cluster_centers_.T, index = s_name, columns = ['Cluster 1', 'Cluster 2','Cluster 3','Cluster 4','Cluster 5',' Cluster 6','Cluster 7'])

from sklearn.metrics import completeness_score, homogeneity_score
c_score = completeness_score(s_class['Value'], kMeans.labels_)
print(c_score)
h_score = homogeneity_score(s_class['Value'], kMeans.labels_)
print(h_score)

#1c.
from sklearn import decomposition
pca = decomposition.PCA(n_components = 4)
dtrans = pca.fit(x_train).transform(x_train)
np.set_printoptions(precision = 2, suppress = True)
print(dtrans)
print(pca.explained_variance_ratio_)

#1d.
kMeans.fit(dtrans)
kMeans.labels_
kMeans.cluster_centers_
pd.DataFrame(kMeans.cluster_centers_.T, index = ['PC 1', 'PC 2', 'PC 3', 'PC 4'], columns = ['Cluster 1', 'Cluster 2','Cluster 3','Cluster 4','Cluster 5',' Cluster 6','Cluster 7'])
c_score = completeness_score(s_class['Value'], kMeans.labels_)
print(c_score)
h_score = homogeneity_score(s_class['Value'], kMeans.labels_)
print(h_score)

#2a.
from apyori import apriori
playlist = pd.read_table("/Users/Yiyang/Documents/CSC 478/playlists/playlists.txt", header = None)
playlist.head()

data = []
for i in open("/Users/Yiyang/Documents/CSC 478/playlists/playlists.txt"):
    t = [int(j) for j in i.split()]
    data.append(t)

print(data[0: 5])

D = data
D

song_df = pd.read_csv("/Users/Yiyang/Documents/CSC 478/playlists/song_names.txt",header = None,sep = '\t', names = ['Index','Song Names'])
song_df.head(5)

song_name = np.array(song_df)
song_name = song_name[:, 1]
song_name

#2b.
#Since Apriori can not be used in Python 3.0 I find a way out from Google,
#The way is going to change the source code of apriori.
#The following code is found from Google
from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
                
    C1.sort()
    return list(map(frozenset, C1))
    

def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k): 
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: 
                retList.append(Lk[i] | Lk[j]) 
    return retList

def apriori(dataSet, minSupport):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def generateRules(L, supportData, metric='confidence', minMetric=0.7):  
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, metric, minMetric)
            else:
                calcMetric(freqSet, H1, supportData, bigRuleList, metric, minMetric)
    return bigRuleList         

def calcMetric(freqSet, H, supportData, brl, metric='confidence', minMetric=0.7):
    prunedH = [] 
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] 
        lift = conf/supportData[conseq] 
        if (metric == 'confidence'):
            if (conf >= minMetric): 
                print (freqSet-conseq,'-->',conseq,'conf:',conf,' lift:',lift)
                brl.append((freqSet-conseq, conseq, conf, lift))
                prunedH.append(conseq)
        elif (metric == 'lift'):
            if (lift >= minMetric): 
                print (freqSet-conseq,'-->',conseq,'conf:',conf,' lift:',lift)
                brl.append((freqSet-conseq, conseq, conf, lift))
                prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, metric='confidence', minMetric=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): 
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcMetric(freqSet, Hmp1, supportData, brl, metric, minMetric)
        if (len(Hmp1) > 1):    
            rulesFromConseq(freqSet, Hmp1, supportData, brl, metric, minMetric)
            
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print (itemMeaning[item])
        print ("           -------->")
        for item in ruleTup[1]:
            print (itemMeaning[item])
        print ("[confidence: %f, lift: %f]" % (ruleTup[2], ruleTup[3]))
        print ()    

L, support = apriori(D, 0.002)


#3a.
import itemBasedRec