# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 15:20:01 2018

@author: cjm
"""
#deal with category variables in the training set
import pandas as pd
import re
import numpy as np
filename = 'train_data.csv'
reviews = pd.read_csv(filename)
cate_train = reviews['categories']
filename = 'testval_data.csv'
reviews = pd.read_csv(filename)
cate_test = reviews['categories']

cate = cate_train.copy()
cate = cate.append(cate_test)

#1. remvoe '[]' in the string
def rem(corpus):
    corpus = re.sub('[\[\]\'\']','',corpus)
    return corpus
cate1 = cate.copy()
cate1 = cate1.apply(rem)
   
#2. split by ',' to find unique categories
global bag
bag = set()
def makecorpus(review):
    review = re.split(',',review)
    global bag
    bag.update(review)
cate2 = cate1.copy()
cate2.apply(makecorpus)

baglist = list(bag)

#3. see counts for each type
from collections import Counter
flatlist = [re.split(',',z) for z in cate1]
flat = [item for z in flatlist for item in z]
counts = Counter(flat) 
cate_count = pd.DataFrame.from_dict(counts, orient='index').reset_index()
cate_count = cate_count.rename(columns={0:'count','index':'category'})
cate_count.to_csv('cate_count.csv', index = False,encoding='utf-8')

#4. get word vectors from the ptr-trained corpus
from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = 'glove.50d.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

size = 50
def review2vec(text, size):
    vec = np.zeros(size).reshape((1, size))
    text = text.lower()
    text = re.split(' ',text)
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

cate_vecs = np.concatenate([review2vec(z, size) for z in baglist])
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(cate_vecs)
klabels = kmeans.labels_
Counter(klabels)
baglist = np.array(baglist)
kindex0 = np.where(klabels==0)

#0:country names:specoal food type
#1:cafe and shopping place
#2:food type
#3:public service school universicty hispital
#4:entertainment golf cinema yoga or small food station

result = pd.DataFrame(klabels)
result.to_csv('categroup.csv', index = False,encoding='utf-8')

    
    
    