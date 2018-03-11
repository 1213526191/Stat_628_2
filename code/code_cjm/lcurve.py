# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 17:40:33 2018

@author: cjm
"""
import gensim 
from gensim.models.deprecated.doc2vec import LabeledSentence
LabeledSentence = LabeledSentence
import numpy as np
from sklearn.preprocessing import scale
import pandas as pd
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM


yelp = pd.read_csv('train_final2.csv')
review = yelp.sample(800000)
review['date'] = scale(review['date'])
from sklearn.cross_validation import train_test_split
train0, test = train_test_split(review,test_size=0.1)

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
        return np.concatenate(vecs)



mse = []
for i in [10000,20000,30000,50000,70000,100000,200000,300000,500000,750000]:
    train = train0[0:i]
    xtrain = list(train['text'])
    x_train = cleanText(xtrain)
    xtest = list(test['text'])
    x_test = cleanText(xtest)
    ntrain = len(xtrain)
    ntest = len(xtest)
    
    x_all = getlabel(x_train, 'TRAIN')
    x_test = getlabel(x_test, 'TEST')
    x_all.extend(x_test)
    x_train = x_all[0:ntrain]
    x_train0 = x_all[0:ntrain]
    x_test0 = x_all[ntrain:]
    
    size = 100
    model = gensim.models.Doc2Vec(dm=0,min_count=5, window=5, size=size, sample=1e-3, negative=5, workers=3)
    model.build_vocab(x_all)
    n=ntrain
    for i in range(20):
        random.shuffle(x_train)
        model.train(x_train,total_examples=n,epochs=1)
    
    trainx = getVecs(model, x_train0, size)
    
    usei = [3]
    usei.extend(list(range(9,17)))
    trainx = np.hstack((trainx,np.reshape(np.array(train.iloc[:,usei]),[train.shape[0],len(usei)])))
    
    for i in range(20):
        random.shuffle(x_test)
        model.train(x_test,total_examples=ntest,epochs=1)
     
    testx = getVecs(model, x_test0, size)
    testx = np.hstack((testx, np.reshape(np.array(test.iloc[:,usei]),[test.shape[0],len(usei)])))
    
    ytrain = np.array(train['stars'])
    ytest = np.array(test['stars'])
    
    
    model = Sequential()
    model.add(LSTM(50,input_shape=(1,109),dropout=0.1))
    model.add(Dense(units=3,activation="sigmoid"))
    model.add(Dense(units=1,activation="linear"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    train0 = trainx.reshape(ntrain,1,109)
    model.fit(train0, ytrain, validation_split=0.3,epochs=10, batch_size=50)
    test0 = testx.reshape(ntest,1,109)
    y_pred = model.predict(test0, batch_size=50)
    y_pred = y_pred.reshape(ntest,)
    
    mse.append(float(sum((y_pred-ytest)**2)/test.shape[0]))

result = pd.DataFrame(mse)
result.to_csv(result,'lcurve.csv',index = False,encoding='utf-8')
