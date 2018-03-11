# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 15:56:22 2018

@author: jianmin
"""

import gensim 
from gensim.models.deprecated.doc2vec import LabeledSentence
LabeledSentence = LabeledSentence

import numpy as np
import pandas as pd
import random
#import re
from sklearn.neural_network import MLPRegressor
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout,Conv1D,MaxPooling1D
from keras.optimizers import SGD
from sklearn.preprocessing import scale

def cleanText(corpus):
    #corpus = [re.sub('[^a-zA-Z]',' ',z) for z in corpus]
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

y_pred = np.array(0)


for i in range(10):
    print(i)
    if i<9:
        j = (i+1)*100000
    else:
        j = 16664+(i+1)*100000
        
    train = pd.read_csv('train_final2.csv')
    train = train.sample(700000,random_state=1)
    train['date'] = scale(train['date'])
    
        
    test = pd.read_csv('test_final2.csv')
    test = test[(100000*i):j]
    test['date'] = scale(test['date'])
       
    x_train = cleanText(list(train['text']))
    ntrain = train.shape[0]
    x_test = cleanText(list(test['text']))
    ntest = test.shape[0]
    
    x_all = getlabel(x_train, 'TRAIN')
    del x_train
    x_test = getlabel(x_test, 'TEST')
    x_all.extend(x_test)
    x_train = x_all[0:ntrain]
    x_train0 = x_all[0:ntrain]
    x_test0 = x_all[ntrain:]
    
    size = 100
    model_dm = gensim.models.Doc2Vec(dm=0,min_count=5, window=5, size=size, sample=1e-3, negative=5, workers=3)
    model_dm.build_vocab(x_all)
    
    for i in range(20):
        random.shuffle(x_all)
        model_dm.train(x_all,total_examples=(ntrain+ntest),epochs=1)
    
    usei = [3]
    usei.extend(list(range(9,17)))
    
    trainx = getVecs(model_dm, x_train0, size)
    trainx = np.hstack((trainx,np.reshape(np.array(train.iloc[:,usei]),[train.shape[0],len(usei)])))
    testx = getVecs(model_dm, x_test0, size)
    testx = np.hstack((testx,np.reshape(np.array(test.iloc[:,usei]),[test.shape[0],len(usei)])))
    
    ytrain = np.array(train['stars'])
    model = Sequential()
    model.add(LSTM(100,input_shape=(1,109),dropout=0.1))
    model.add(Dense(units=3,activation="sigmoid"))
    model.add(Dense(units=1,activation="linear"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    model.fit(trainx.reshape(ntrain,1,109), ytrain, validation_split=0.3,epochs=5, batch_size=50)
    test0 = testx.reshape(ntest,1,109)
    predi = model.predict(test0, batch_size=50)
    predi = predi.reshape(ntest,)
    

    y_pred = np.append(y_pred,predi)
    
        
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('pre80w1005.csv', index = False,encoding='utf-8')       