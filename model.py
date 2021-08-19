import numpy as np ## scientific computation
import pandas as pd ## loading dataset file
##import matplotlib.pyplot as plt ## Visulization

#select data source
data= pd.read_csv('C:/PYDATAFILES/MOVIES.csv', encoding='latin')
#output the shape of the data container(frame)
print(data)

print(data.shape)  ### Return the shape of data 
print(data.ndim)   ### Return the n dimensions of data
print(data.size)   ### Return the size of data 
print(data.isna().sum())  ### Returns the sum fo all na values
print(data.info())  ### Give concise summary of a DataFrame
print(data.head())  ## top 5 rows of the dataframe
print(data.tail()) ## bottom 5 rows of the dataframe

#import nltk  ## Preprocessing Reviews
import nltk  ## Preprocessing Reviews
#nltk.download('stopwords') ##Downloading stopwords
#nltk.download('wordnet')
from nltk.corpus import stopwords ## removing all the stop words
from nltk.stem.porter import PorterStemmer ## stemming of words
from nltk.stem import WordNetLemmatizer
import re  ## To use Regular expression
stemmer = PorterStemmer()



corpus = []
for i in range(len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
    review = review.lower()
    review = review.split()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not') 
    #remove negative word 'not' as it is closest word to help determine whether the review is good or not 
    review = [stemmer.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

print(corpus)
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000) ##1500 columns
X = cv.fit_transform(corpus).toarray()

y = data["Sentiment"]

import pickle
# Creating a pickle file for the CountVectorizer
pickle.dump(cv, open('countvector.pkl', 'wb'))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)



#GaussianNB
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#classifier = GaussianNB().fit(X_train, y_train)
#MNB = MultinomialNB()
#cls = MultinomialNB().fit(X_train, y_train)
gnb = GaussianNB(var_smoothing=1e-2)
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=12)
cv = cross_val_score(gnb,X_train,y_train,cv=kfold)
print(cv)
print(cv.mean()*100)
gnb.fit(X_train,y_train)
y_pred_gnb=gnb.predict(X_test)
print('The accuracy of the Naive Bayes is', metrics.accuracy_score(y_pred_gnb,y_test)*100)
cm=confusion_matrix(y_test, y_pred_gnb)
print(cm)

#MultinomialNB
mnb = MultinomialNB(alpha=2)
cv = cross_val_score(mnb,X_train,y_train,cv=kfold)
print(cv)
print(cv.mean()*100)
mnb.fit(X_train,y_train)
y_pred_mnb=mnb.predict(X_test)
print('The accuracy of the Naive Bayes is', metrics.accuracy_score(y_pred_mnb,y_test)*100)
cm=confusion_matrix(y_test, y_pred_mnb)
print(cm)

# Creating a pickle file for the Multinomial Naive Bayes model
#filename = 'voting_clf.pkl'
pickle.dump(mnb, open("Review.pkl", 'wb'))