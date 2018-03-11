# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 16:23:06 2018

@author: cjm
"""
import gensim 
from gensim.models.deprecated.doc2vec import LabeledSentence
LabeledSentence = LabeledSentence
import numpy as np
from sklearn.preprocessing import scale
import pandas as pd
from sklearn import linear_model

yelp = pd.read_csv('ran1wsenti.csv')
review = yelp
review['date'] = scale(review['date'])

def clean(corpus):
    corpus = [z.split() for z in corpus]
    return corpus

def gettab(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


from sklearn.cross_validation import train_test_split
train, test = train_test_split(review,test_size=0.4)

xtrain = list(train['text'])
x_train = clean(xtrain)
xtest = list(test['text'])
x_test = clean(xtest)
ntrain = len(xtrain)
ntest = len(xtest)

x_all = gettab(x_train, 'TRAIN')
x_test = gettab(x_test, 'TEST')
x_all.extend(x_test)
x_train = x_all[0:ntrain]
x_train0 = x_all[0:ntrain]
x_test0 = x_all[ntrain:]

#train the doc2vec model to get sentence vectors
import random
size = 100
 
model = gensim.models.Doc2Vec(dm=0,min_count=5, window=5, size=size, sample=1e-3, negative=5, workers=3)
model.build_vocab(x_all)
n=ntrain
 
for i in range(20):
    random.shuffle(x_train)
    model.train(x_train,total_examples=n,epochs=1)
    
def getvecs(model, corpus, size):
    vecs = [np.array(model[z[1]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)
 
trainx1 = getvecs(model, x_train0, size)

usei = [3]
usei.extend(list(range(10,17)))
trainall = np.hstack((trainx1,np.reshape(np.array(train.iloc[:,usei]),[train.shape[0],len(usei)])))

for i in range(20):
    random.shuffle(x_test)
    model.train(x_test,total_examples=ntest,epochs=1)
    
testx1 = getvecs(model, x_test0, size)
testall = np.hstack((testx1, np.reshape(np.array(test.iloc[:,usei]),[test.shape[0],len(usei)])))

vector=[]
ad=[]
vad=[]

ytrain = np.array(train['stars'])
ytest = np.array(test['stars'])


################################################
regr = linear_model.LinearRegression()
regr.fit(trainx1, ytrain)
testpred = regr.predict(testx1)
vector.append(float(sum((testpred-ytest)**2))/test.shape[0])

regr = linear_model.LinearRegression()
regr.fit(trainall, ytrain)
testpred = regr.predict(testall)
vad.append(float(sum((testpred-ytest)**2))/test.shape[0])

adtrain= np.array(train.iloc[:,usei])
adtest = np.array(test.iloc[:,usei])
regr = linear_model.LinearRegression()
regr.fit(adtrain, ytrain)
testpred = regr.predict(adtest)
ad.append(float(sum((testpred-ytest)**2))/test.shape[0])

#############################################
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(trainx1, ytrain).predict(testx1)
vector.append(float(sum((y_pred-ytest)**2))/test.shape[0])

gnb = GaussianNB()
y_pred = gnb.fit(trainall, ytrain).predict(testall)
vad.append(float(sum((y_pred-ytest)**2))/test.shape[0])

gnb = GaussianNB()
y_pred = gnb.fit(adtrain, ytrain).predict(adtest)
ad.append(float(sum((y_pred-ytest)**2))/test.shape[0])

##################################################

#svr: not good
from sklearn.svm import SVR
clf = SVR(C=1.0, epsilon=0.2,gamma=0.1)
clf.fit(adtrain, ytrain) 
y_pred = clf.predict(adtest)
float(sum((y_pred-ytest)**2)/test.shape[0])

####################################################

from sklearn.neural_network import MLPRegressor
clf = MLPRegressor(solver='adam', activation='logistic',alpha=1e-4,hidden_layer_sizes=(10), random_state=1)
clf.fit(trainx1, ytrain)
y_pred = clf.predict(testx1)
vector.append(float(sum((y_pred-ytest)**2))/test.shape[0])

clf = MLPRegressor(solver='adam', activation='logistic',alpha=1e-4,hidden_layer_sizes=(10), random_state=1)
clf.fit(trainall, ytrain)
y_pred = clf.predict(testall)
vad.append(float(sum((y_pred-ytest)**2))/test.shape[0])

clf = MLPRegressor(solver='adam', activation='logistic',alpha=1e-4,hidden_layer_sizes=(10), random_state=1)
clf.fit(adtrain, ytrain)
y_pred = clf.predict(adtest)
ad.append(float(sum((y_pred-ytest)**2))/test.shape[0])

#######################################################

### LSTM: final model
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(50,input_shape=(1,100),dropout=0.1))
model.add(Dense(units=3,activation="sigmoid"))
model.add(Dense(units=1,activation="linear"))
model.compile(loss='mean_squared_error', optimizer='adam')

train0 = trainx1.reshape(ntrain,1,100)
model.fit(train0, ytrain, validation_split=0.3,epochs=10, batch_size=50)
test0 = testx1.reshape(ntest,1,100)
y_pred = model.predict(test0, batch_size=50)
y_pred = y_pred.reshape(ntest,)
vector.append(float(sum((y_pred-ytest)**2))/test.shape[0])

model = Sequential()
model.add(LSTM(50,input_shape=(108,1),dropout=0.1))
model.add(Dense(units=3,activation="sigmoid"))
model.add(Dense(units=1,activation="linear"))
model.compile(loss='mean_squared_error', optimizer='adam')

train0 = trainall.reshape(ntrain,108,1)
model.fit(train0, ytrain, validation_split=0.3,epochs=10, batch_size=50)
test0 = testall.reshape(ntest,108,1)
y_pred = model.predict(test0, batch_size=50)
y_pred = y_pred.reshape(ntest,)
vad.append(float(sum((y_pred-ytest)**2))/test.shape[0])

model = Sequential()
model.add(LSTM(50,input_shape=(1,8),dropout=0.1))
model.add(LSTM(50,dropout=0.1))
model.add(Dense(units=3,activation="sigmoid"))
model.add(Dense(units=1,activation="linear"))
model.compile(loss='mean_squared_error', optimizer='adam')

train0 = adtrain.reshape(ntrain,1,8)
model.fit(train0, ytrain, validation_split=0.3,epochs=10, batch_size=50)
test0 = adtest.reshape(ntest,1,8)
y_pred = model.predict(test0, batch_size=50)
y_pred = y_pred.reshape(ntest,)
ad.append(float(sum((y_pred-ytest)**2))/test.shape[0])

###################################################
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(trainx1, ytrain)
y_pred = mul_lr.predict(testx1)  
vector.append(float(sum((y_pred-ytest)**2))/test.shape[0])

mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(trainall, ytrain)
y_pred = mul_lr.predict(testall)  
vad.append(float(sum((y_pred-ytest)**2))/test.shape[0])

mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(adtrain, ytrain)
y_pred = mul_lr.predict(adtest)  
ad.append(float(sum((y_pred-ytest)**2))/test.shape[0])

result = pd.DataFrame(columns={'v':vector,'ad':ad,'vad':vad})
result.to_csv('result.csv')
