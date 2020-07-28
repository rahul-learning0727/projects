# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:06:11 2020

@author: RJ PC
"""

import pandas as pd
messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message']) 

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

corpus = []
for i in range(0, len(messages)):
    review = re.sub('^a-zA-Z',' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word)for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = CountVectorizer(max_features=5000)
#x = vectorizer.fit_transform(corpus).toarray()
# with TFID

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(corpus).toarray()


y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train,y_train)

y_pred = spam_detect_model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)*100
