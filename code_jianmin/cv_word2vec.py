# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 20:12:40 2018

@author: cjm
"""
from gensim.models import KeyedVectors

import numpy as np
import pandas as pd
yelp = pd.read_csv('chen_first100000.csv')
review = yelp[['stars','text']]

#get index for cv
import random
index = list(range(review.shape[0]))
random.seed(1)
random.shuffle(index)
def getcvi(index,k):
    n = len(index)
    l = int(n/k)
    use = []
    for i in range(k-1):
        use.append(index[l*i:(l*i+l)])
    use.append(index[l*(k-1):])
    return use
index = getcvi(index,5)

def cleanText(corpus):
    corpus = [z.split() for z in corpus]
    return corpus

#get sentence vectors from the ptr-trained corpus
def review2vec(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


#####cv for OLS
err = 0
from sklearn import linear_model
from sklearn.preprocessing import scale
model = KeyedVectors('200dtwittermodel')
for i in range(5):
    itest = review.index.isin([index[i]])
    train = review[~itest]
    test = review.iloc[index[i],:]
    
    xtrain = list(train['text'])
    x_train = cleanText(xtrain)
    xtest = list(test['text'])
    x_test = cleanText(xtest)
        
    n_dim = 200
    train_vecs = np.concatenate([review2vec(z, n_dim) for z in x_train])
    train_vecs = scale(train_vecs)
    test_vecs = np.concatenate([review2vec(z, n_dim) for z in x_test])
    test_vecs = scale(test_vecs)
    
    ytrain = np.array(train['stars'])
    ytest = np.array(test['stars'])
    regr = linear_model.LinearRegression()
    regr.fit(train_vecs, ytrain)
    testpred = regr.predict(test_vecs)
    
    err += sum((testpred-ytest)**2)

print(err)