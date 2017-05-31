import re
import sys
import json
import time
from sklearn.datasets.base import Bunch
from collections import OrderedDict
from glob import glob
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import os
from itertools import izip
import scraper
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
########################################################      Data preprocessing       ########################################

values=[]
all_data=OrderedDict()
categories = []
path='articles/'


categories=['business','entertainment','health','politics','science','sports','technology']

for i in xrange(7):
	path1=path+categories[i]+'.json'
	with open(path1,'r') as jsonfile:
		data=json.loads(jsonfile.read())
		#print(data)
		all_data[categories[i]]=data
	best_category = min([len(c) for c in all_data.values()])
#	print(best_category)
	values = [[j] * best_category for j in xrange(len(categories))]
#	print(values)

join=lambda x,y :x+y
data_train=Bunch(categories=categories,
                 values=reduce(join, values),
                 data=reduce(join, [c[:best_category] for c in all_data.values()]))


new_data=''
stop = set(stopwords.words('english'))
stopped_data=[i for i in data_train.data[0].lower().split() if i not in stop]
for i in stopped_data:
	new_data=new_data+' '+i
print(new_data)
token=nltk.word_tokenize(new_data)
bigrams=ngrams(token,2)
trigrams=ngrams(token,3)
print(list(bigrams),list(trigrams))

##############################################     Feature Selection and Training      #####################################



												#Trigrams
bigram_vectorizer = CountVectorizer(ngram_range=(2, 3),
                                     token_pattern=r'\b\w+\b', min_df=1)

												#Term frequency-inverse document frequency
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english', max_features=6000,
                                 strip_accents='unicode')

data_weighted = vectorizer.fit_transform(data_train.data)

data_weighted=bigram_vectorizer.fit_transform(data_train.data)

#print(data_train.data)

#feature_selection = SelectPercentile(f_classif, percentile=20)

#data_weighted = feature_selection.fit_transform(data_weighted, data_train['values'])

clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)					#Support Vector Machine
clf.fit(data_weighted, data_train['values'])

if not os.path.exists('training'):
        os.mkdir('training')

filename = 'training/{0}.pkl'.format(int(time.time()))
joblib.dump({'clf': clf,
                 'vectorizer': vectorizer,
		 'countvectorizer':bigram_vectorizer,
                 }, filename, compress=9)

'''
prediction = clf.predict(data_weighted)

for j in xrange(7):
	path2=path+categories[i]+'.json'
	with open(path2,'r') as jsonfile:
		data=json.loads(jsonfile.read())
		for article in data.get('articles'):
			all_data.extend([scraper.clean(article['content'])])

data=Bunch(categories=scraper.CATEGORIES.keys(),
                 values=None,
                 data=all_data)

print(prediction)
for text, prediction in izip(data.data, prediction):
        print (scraper.CATEGORIES.keys()[prediction].ljust(15, ' '), text[:100], '...')

'''
