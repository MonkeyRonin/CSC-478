#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

bank_df = pd.read_csv("/Users/Yiyang/Documents/CSC 478/bank_data.csv", sep = ',')
bank_df

bank_df.shape
bank_df.head(10)
bank_df.columns
bank_df.dtypes

#1. Statistics associated with numerical
bank_df.describe()
#1. distributions of value associated with categorical
bank_df.describe(include = "all")

#2. Subsets
pep_y_df = bank_df[bank_df["pep"] == "YES"]
pep_y_df.head(10)
pep_y_df.describe(include = "all")

pep_n_df = bank_df[bank_df["pep"] == "NO"]
pep_n_df.head(10)
pep_n_df.describe(include = "all")

#3. z-score normalization
income_z = (bank_df["income"] - bank_df["income"].mean())/bank_df["income"].std()
income_z.head(5)

#4. Categorical age
age_bins = pd.qcut(bank_df.age, [0, 0.33, 0.66, 1], labels = ["young", "mid-age", "old"])
age_bins

#5. Max-min normalization
bank_df["income"] = (bank_df["income"] - bank_df["income"].min())/(bank_df["income"].max() - bank_df["income"].min())
bank_df["age"] = (bank_df["age"] - bank_df["age"].min())/(bank_df["age"].max() - bank_df["age"].min())
bank_df["children"] = (bank_df["children"] - bank_df["children"].min())/(bank_df["children"].max() - bank_df["children"].min())

#6. Dummy
bank_dm = pd.get_dummies(bank_df)
bank_dm.head(10)
bank_dm.to_csv("/Users/Yiyang/Documents/CSC 478/bank_numeric.csv", float_format = "%1.2f")

#7. Correlation Analysis
bank_df_new = bank_df.drop('id', axis = 1)
bank_dm_new = pd.get_dummies(bank_df_new)
bank_dm_new.head(10)
bank_dm_new.corr()

#8. Scatterplot
bank_df.plot(x = "income", y = "age", kind = "scatter")

#9. Histogram
bank_df["income"].plot(kind = "hist", bins = 9)
bank_df["age"].plot(kind = "hist", bins = 15)

#10. bargraph
region_dis = bank_df["region"].value_counts()/bank_df["region"].count()

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(121)
ax.set_xlabel('Region')
ax.set_ylabel('Percentage')
ax.set_title("Region Distribution")
region_dis.plot(kind='bar', grid = True)

#11. Cross-tabulation
bank_df.groupby(["region", "pep"])["pep"].count()
rp = pd.crosstab(bank_df["region"], bank_df["pep"])
rp

plt.show(rp.plot(kind = "bar"))