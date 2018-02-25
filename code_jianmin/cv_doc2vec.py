# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 19:31:35 2018

@author: cjm
"""
import gensim 
from gensim.models.deprecated.doc2vec import LabeledSentence
LabeledSentence = LabeledSentence

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

def getlabel(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

def getVecs(model, corpus, size):
    vecs = [np.array(model[z[1]]).reshape((1, size)) for z in corpus]
    #if vecs == 0:
    #    vecs = np.zeros([1,size])
    return np.concatenate(vecs)

#####cv for OLS
err = 0
from sklearn import linear_model
for i in range(5):
    itest = review.index.isin([index[i]])
    train = review[~itest]
    test = review.iloc[index[i],:]
    
    xtrain = list(train['text'])
    x_train = cleanText(xtrain)
    xtest = list(test['text'])
    x_test = cleanText(xtest)
    ntrain = train.shape[0]
    ntest = test.shape[0]
    
    x_all = getlabel(x_train, 'TRAIN')
    x_test = getlabel(x_test, 'TEST')
    x_all.extend(x_test)
    x_train = x_all[0:ntrain]
    x_train0 = x_all[0:ntrain]
    x_test0 = x_all[ntrain:]
    
    size = 300
    model_dm = gensim.models.Doc2Vec(dm=0,min_count=5, window=5, size=size, sample=1e-3, negative=5, workers=3)
    model_dm.build_vocab(x_all)
    model_dm.train(x_train,total_examples=ntrain,epochs=1)
    for i in range(20):
        random.shuffle(x_train)
        model_dm.train(x_train,total_examples=ntrain,epochs=1)
    
    train_vecs_dm = getVecs(model_dm, x_train0, size)
    for i in range(20):
        random.shuffle(x_test)
        model_dm.train(x_test,total_examples=ntest,epochs=1)
    test_vecs_dm = getVecs(model_dm, x_test0, size)
    del model_dm

    ytrain = np.array(train['stars'])
    ytest = np.array(test['stars'])
    regr = linear_model.LinearRegression()
    regr.fit(train_vecs_dm, ytrain)
    testpred = regr.predict(test_vecs_dm)
    
    err += sum((testpred-ytest)**2)
    
print(err)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    