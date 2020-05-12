#!/usr/bin/env python
# coding: utf-8

# Bank Marketing Data Set
# ---
# Podaci se odnose na direktne marketinške kampanje (telefonski pozivi) portugalske banke. Cilj klasifikacije je da predvidi da li će se klijent pretplatiti na oročeni depozit.

# In[1]:


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ### Priprema i analiza podataka
# ---

# In[2]:


dataset = pd.read_csv('bank-full.csv', sep = ';')
dataset.replace(('yes', 'no'), (1, 0), True)

benchmark = False
if not benchmark:
    dataset = dataset.drop('duration', axis = 1)
    
dataset.head(11)


# In[3]:


dataset.describe()


# In[4]:


ages, x = np.arange(3), np.arange(3)
for age in dataset['age']:
    if age >= 18 and age <= 44:
        ages[0] += 1
    elif age >= 45 and age <= 65:
        ages[1] += 1
    elif age >= 66:
        ages[2] += 1

plt.figure(figsize = (7, 5))
sns.barplot(x, ages)
plt.xticks(x, ('odrasle osobe 18-44', 'srednjih godina 45-65', 'stari > 65'))
plt.xlabel('Starostna grupa')
plt.ylabel('Broj')
plt.show()


# In[5]:


plt.figure(figsize = (15, 5))
sns.countplot(x = 'job', data = dataset)
plt.xlabel('Posao')
plt.ylabel('Broj')
plt.show()


# In[6]:


sns.countplot(x = 'default', data = dataset)
plt.xticks(x, ('Da', 'Ne'))
plt.xlabel('Kreditno zaduzenje')
plt.ylabel('Broj')
plt.show()


# In[7]:


sns.countplot(x = 'housing', data = dataset)
plt.xticks(x, ('Da', 'Ne'))
plt.xlabel('Stambeni kredit')
plt.ylabel('Broj')
plt.show()


# In[8]:


sns.countplot(x = 'loan', data = dataset)
plt.xticks(x, ('Da', 'Ne'))
plt.xlabel('Licni zajam')
plt.ylabel('Broj')
plt.show()


# ### Korelacije atributa
# ---

# In[9]:


correlations = dataset.corr()

plt.figure(figsize = (15, 10))
sns.heatmap(correlations, annot = True)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.title('Korelacije atributa')
plt.show()


# ### Priprema seta za trening i predikciju
# ---

# In[10]:


cat_features = [[feature, .0] for feature in list(dataset.select_dtypes(object).columns)]
num_features = [[feature, .0] for feature in list(dataset.select_dtypes(exclude = object).columns)]
del num_features[-1]
first_run = True # prvi put pokrecemo skriptu kada su vaznosti num i cat features == .0

dataset_train, dataset_test = train_test_split(dataset, test_size = .1, random_state = 42)

dataset_test_original = dataset_test.copy()

dataset_train = pd.get_dummies(dataset_train, columns = dataset_train.select_dtypes(object).columns)
dataset_test = pd.get_dummies(dataset_test, columns = dataset_test.select_dtypes(object).columns)

y = pd.factorize(dataset_train['y'])[0]
dataset_train = dataset_train.drop('y', axis = 1)

dataset_test = dataset_test.drop('y', axis = 1)


# ### Trening
# ---

# In[11]:


clf = RandomForestClassifier(warm_start = True, oob_score = True)

min_estimators = 100
max_estimators = 2000

oob_errors = []
min_oob_error = 1.0
min_oob_estimator = min_estimators

plt.figure(figsize = (15, 5))
for index in range(min_estimators, max_estimators + 1, 100):
    clf.set_params(n_estimators = index)
    clf.fit(dataset_train, y)
    
    oob_error = 1 - clf.oob_score_
    oob_errors.append((index, oob_error))
    
    if (oob_error < min_oob_error):
        min_oob_error = oob_error
        min_oob_estimator = index
        importances = clf.feature_importances_

    xoob, yoob = zip(*oob_errors)
    plt.plot(xoob, yoob)
    
plt.plot(min_oob_estimator, min_oob_error, 'o')
plt.xlim(min_estimators, max_estimators)
plt.title('Stopa OOB gresaka')
plt.xlabel('broj estimatora')
plt.ylabel('stopa greske')


# ### Izracunavanje vaznosti kategorijalnih i numerickih atributa
# ---

# In[12]:


if first_run:
    index, cat_feature_index, num_feature_index = (0,) * 3
    last_cat_feature = cat_features[0][0]

    for feature in list(dataset_train):
        if feature.startswith(tuple(cat_feature.rsplit('_', 1)[0] for cat_feature, importance in cat_features)):
            if not feature.startswith(last_cat_feature):
                last_cat_feature = feature.rsplit('_', 1)[0]
                cat_feature_index += 1
                
            cat_features[cat_feature_index][1] += importances[index]
        else:
            num_features[num_feature_index][1] = importances[index]
            num_feature_index += 1
            
        index += 1
        
    feature_importances = num_features + cat_features
    
    first_run = False
    
plt.figure(figsize = (15, 5))
xfi, yfi = zip(*feature_importances)
plt.bar(xfi, yfi)
plt.xticks(rotation = 90)
plt.title('Vaznost atributa')
plt.xlabel('ime atributa')
plt.ylabel('vaznost')


# ### Predikcija
# ---

# In[13]:


test_y = clf.predict(dataset_test)

dataset_test_original['y'] = test_y

dataset_test_original.loc[dataset_test_original['y'] == 1].head(10)


# ### Analiza predikcije gde je y = 1 (pretplatio se na oročeni depozit)
# ---

# In[14]:


dataset_deposited = dataset_test_original.loc[dataset_test_original['y'] == 1]


# In[15]:


ages, x = np.arange(3), np.arange(3)
for age in dataset_deposited['age']:
    if age >= 18 and age <= 44:
        ages[0] += 1
    elif age >= 45 and age <= 65:
        ages[1] += 1
    elif age >= 66:
        ages[2] += 1

plt.figure(figsize = (7, 5))
sns.barplot(x, ages)
plt.xticks(x, ('odrasle osobe 18-44', 'srednjih godina 45-65', 'stari > 65'))
plt.xlabel('Starostna grupa')
plt.ylabel('Broj')
plt.show()


# In[16]:


plt.figure(figsize = (15, 5))
sns.countplot(x = 'job', data = dataset_deposited)
plt.xlabel('Posao')
plt.ylabel('Broj')
plt.show()


# In[17]:


sns.countplot(x = 'default', data = dataset_deposited)
plt.xticks(x, ('Da', 'Ne'))
plt.xlabel('Kreditno zaduzenje')
plt.ylabel('Broj')
plt.show()


# In[18]:


sns.countplot(x = 'housing', data = dataset_deposited)
plt.xticks(x, ('Da', 'Ne'))
plt.xlabel('Stambeni kredit')
plt.ylabel('Broj')
plt.show()


# In[19]:


sns.countplot(x = 'loan', data = dataset_deposited)
plt.xticks(x, ('Da', 'Ne'))
plt.xlabel('Licni zajam')
plt.ylabel('Broj')
plt.show()

