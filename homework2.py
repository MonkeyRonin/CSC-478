#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 19:29:19 2017

@author: Yiyang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#1.
train_mm = pd.read_table("/Users/Yiyang/Documents/CSC 478/newsgroups/trainMatrixModified.txt",header=None)
train_mm.shape
test_mm = pd.read_table("/Users/Yiyang/Documents/CSC 478/newsgroups/testMatrixModified.txt",header=None)
test_mm.head()

mterm = pd.read_table("/Users/Yiyang/Documents/CSC 478/newsgroups/modifiedterms.txt",header=None)
mterm.head()

train_class = pd.read_table("/Users/Yiyang/Documents/CSC 478/newsgroups/trainClasses.txt",header=None)
train_class.head()

train_label = train_class.drop(train_class.columns[0], axis = 1)
train_label.head()

test_class = pd.read_table("/Users/Yiyang/Documents/CSC 478/newsgroups/testClasses.txt",header=None)
test_class.head()

test_label = test_class.drop(test_class.columns[0], axis = 1)
test_label

train_dt = train_mm.T
test_dt = test_mm.T

train_dt

term = np.genfromtxt("/Users/Yiyang/Documents/CSC 478/newsgroups/testClassesmodifiedterms.txt", dtype=str)
term[0:30]

tf = train_dt.sum(axis = 1)
dict_tf = {}
for i in range(len(tf)):
    dict_tf[term[i] = tf[1]]
print(sorted(dict_tf.items()))
sdict_tf = sorted(sdict_tf.value(), reverse = True)

#a.
train_dt = np.array(train_dt)
test_dt = np.array(test_dt)
train_label = np.array(train_label)
test_label = np.array(test_label)

def searchknn(x, d, l, k, m):
    if m == 0:
        index = x
        diffMat = np.tile(index, (d.shape[0], 1)) - d
        sqDiffMat = diffMat**2
        sqDistance = sqDiffMat.sum(axis = 1)
        distance = sqDistance**0.5
    elif m == 1:
        dnorm = np.array([np.linalg.norm(d[i]) for i in range(len(d))])
        xnorm = np.linalg.norm(x)
        s = np.dot(d, x)/(dnorm * xnorm)
        distance = 1 - s
    index = np.argsort(distance)
    n_label = l[index[:k]]
    f0 = 0
    f1 = 0
    for inx in n_label:
        if l[inx] == 0:
            f0 += 1
        else:
            f1 += 1
        if f0 < f1:
            pred_class = 1
        else:
            pred_class = 0
    return index[:k], pred_class

n_index, pred_class = searchknn(test_dt[1], train_dt, train_label, 9, 1)
n_index
pred_class

#b.
def daccuracy(x, d, trainl, testl, k, m):
    count = 0
    correct = 0
    for i in range(x.shape[0]):
        count += 1
        n_index, pred_class = searchknn(x[i,:], d, trainl, k, m)
        if pred_class == testl[i]:
            correct += 1
    acc = correct/count
    return acc

acc = daccuracy(test_dt, train_dt, train_label, test_label, 5, 0)
acc

#c.
def accplot(x, d, trainl, testl):
    i = 1
    EucliAcc = {}
    CosineAcc = {}
    for i in range(20):
        i += 1
        EucliAcc[i] = daccuracy(x, d, trainl, testl, i, 0)
        CosineAcc[i] = daccuracy(x, d, trainl, testl, i, 1)
    df_eucli = pd.DataFrame(list(EucliAcc.items()))
    df_cosine = pd.DataFrame(list(CosineAcc.items()))
    
accplot(test_dt, train_dt, train_label, test_label)
    
#d.
train_mm = pd.read_table("/Users/Yiyang/Documents/CSC 478/newsgroups/trainMatrixModified.txt",header=None)
test_mm= pd.read_table("/Users/Yiyang/Documents/CSC 478/newsgroups/testMatrixModified.txt",header=None)

train_td = np.array(train_mm)
test_td = np.array(test_mm)
num_traint = len(train_td[:,0])
n_train = len(train_td[0])
print(num_traint)
print(n_train)

num_testt = len(test_td[:,0])
n_test = len(test_td[0])
print(num_testt)
print(n_test)

train_df = np.array([(train_td != 0).sum(1)]).T
test_df = np.array([(test_td != 0).sum(1)]).T

print(train_df)
print(test_df)

train_matrix = np.ones(np.shape(train_td), dtype = float) * n_train
np.set_printoptions(precision = 2, suppress = True, linewidth = 120)
print(train_matrix)

test_matrix = np.ones(np.shape(test_td), dtype = float) * n_test
np.set_printoptions(precision = 2, suppress = True, linewidth = 120)
print(test_matrix)
    
#2.
#a
bd_df = pd.read_csv("/Users/Yiyang/Documents/CSC 478/bank_data.csv", index_col = 0)
bd_df.shape
bd_df.head()

bd_records = bd_df[['age', 'income', 'children', 'gender', 'region', 'married', 'car', 'savings_acct', 'current_acct', 'mortgage']]
bd_records.head()

bd_target = bd_df.pep
bd_target.head()

bd_dm = pd.get_dummies(bd_records[['age', 'income', 'children', 'gender', 'region', 'married', 'car', 'savings_acct', 'current_acct', 'mortgage']])
bd_dm.head(10)

from sklearn.cross_validation import train_test_split
bd_train, bd_test, bd_target_train, bd_target_test = train_test_split(bd_dm, bd_target, test_size=0.2, random_state=33)

np.set_printoptions(precision = 4, linewidth = 80, suppress = True)
print(bd_test[0: 5])
print(bd_train[0: 5])

#b.
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler().fit(bd_train)
bd_train_norm = min_max_scaler.transform(bd_train)
bd_test_norm = min_max_scaler.transform(bd_test)

np.set_printoptions(precision=2, linewidth=80, suppress=True)
bd_train_norm[0: 5]
bd_test_norm[0: 5]

from sklearn import neighbors, tree, naive_bayes
nn = 5
knnclf = neighbors.KNeighborsClassifier(nn, weights = 'distance')
knnclf.fit(bd_train_norm, bd_target_train)

knnpreds_test = knnclf.predict(bd_test_norm)
print(knnpreds_test)

from sklearn.metrics import classification_report
print(classification_report(bd_target_test, knnpreds_test))

from sklearn.metrics import confusion_matrix
knncm = confusion_matrix(bd_target_test, knnpreds_test)
print(knncm)

print(knnclf.score(bd_test_norm, bd_target_test))
print(knnclf.score(bd_train_norm, bd_target_train))

#c. Decision Tree
treeclf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=3)
treeclf = treeclf.fit(bd_train, bd_target_train)
treepreds_test = treeclf.predict(bd_test)
print(treepreds_test)
print(treeclf.score(bd_test, bd_target_test))
print(treeclf.score(bd_train, bd_target_train))
print(classification_report(bd_target_test, treepreds_test))
treecm = confusion_matrix(bd_target_test, treepreds_test, labels = ['Yes','No'])
print(treecm)

#c. Naive Bayes 
nbclf = naive_bayes.GaussianNB()
nbclf = nbclf.fit(bd_train, bd_target_train)
nbpreds_test = nbclf.predict(bd_test)
print(nbpreds_test)

print(nbclf.score(bd_train, bd_target_train))
print(nbclf.score(bd_test, bd_target_test))


#3.
#Remove N/A
am_df = pd.read_csv("/Users/Yiyang/Documents/CSC 478/adult-modified.csv", sep = ",", na_values = ["?"])
am_df.shape
am_df.head()

am_df[am_df.age.isnull()]

age_mean = am_df.age.mean()
am_df.age.fillna(age_mean, axis = 0, inplace = True)

am_df.dropna(axis = 0, inplace = True)
am_df.shape

am_df.describe(include = "all")

#Age Histogram
plt.hist(am_df.age)
plt.title("Age Histogram")
plt.xlabel("Age")
plt.ylabel("Frequency")

#Education Hist
plt.hist(am_df.education)
plt.title("Education Histogram")
plt.xlabel("Education")
plt.ylabel("Frequency")

#Hours per week Hist
plt.hist("hours-per-week", data = am_df)
plt.title("Hours Histogram")
plt.xlabel("Hours")
plt.ylabel("Frequency")

#Work Class Bar Chart
wc_dis = am_df["workclass"].value_counts()/am_df["workclass"].count()
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(121)
ax.set_xlabel('Work Class')
ax.set_ylabel('Percentage')
ax.set_title("Work Class Distribution")
wc_dis.plot(kind='bar', grid = True)

#Marital Status Bar
ms_dis = am_df["marital-status"].value_counts()/am_df["marital-status"].count()
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(121)
ax.set_xlabel('Marital Status')
ax.set_ylabel('Percentage')
ax.set_title("Marital Status Distribution")
ms_dis.plot(kind='bar', grid = True)

#Race Bar
race_dis = am_df["race"].value_counts()/am_df["race"].count()
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(121)
ax.set_xlabel('Race')
ax.set_ylabel('Percentage')
ax.set_title("Race Distribution")
race_dis.plot(kind='bar', grid = True)

#Income Bar
income_dis = am_df["income"].value_counts()/am_df["income"].count()
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(121)
ax.set_xlabel('Income')
ax.set_ylabel('Percentage')
ax.set_title("Income Distribution")
income_dis.plot(kind='bar', grid = True)

#Sex
sex_dis = am_df["sex"].value_counts()/am_df["sex"].count()
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(121)
ax.set_xlabel('Sex')
ax.set_ylabel('Percentage')
ax.set_title("Sex Distribution")
sex_dis.plot(kind='bar', grid = True)

#Cross Tabulation
#Education + Race
am_df.groupby(["education", "race"])["race"].count()
rp = pd.crosstab(am_df["education"], am_df["race"])
plt.show(rp.plot(kind = "bar"))

#WorkClass + income
am_df.groupby(["workclass", "income"])["income"].count()
rp = pd.crosstab(am_df["workclass"], am_df["income"])
plt.show(rp.plot(kind = "bar"))

#Work-class + Race 
am_df.groupby(["workclass", "race"])["race"].count()
rp = pd.crosstab(am_df["workclass"], am_df["race"])
plt.show(rp.plot(kind = "bar"))

#Race + Income
am_df.groupby(["race", "income"])["income"].count()
rp = pd.crosstab(am_df["race"], am_df["income"])
plt.show(rp.plot(kind = "bar"))
#Percentage Table
pd.crosstab(am_df.race, am_df.income).apply(lambda r: r/r.sum(), axis=1)

