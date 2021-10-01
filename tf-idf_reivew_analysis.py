#!/usr/bin/env python
# coding: utf-8
# In[141]:
from os import sep
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
import re
from bs4 import BeautifulSoup
import bs4
import contractions
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report
# In[2]:
# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz
# ## Read Data from S3
# In[24]:
#data = pd.read_csv('amazon_reviews_us_Kitchen_v1_00.tsv', sep='\t', usecols=['star_rating','review_body'])
df = pd.read_csv('https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz', compression='gzip', sep='\t', warn_bad_lines=False, error_bad_lines=False)
# ## Keep Reviews and Ratings
# In[47]:
df = df.dropna(axis = 0)
df = df[['star_rating','review_body']]
# In[48]:
# # Labelling Reviews:
# ## The reviews with rating 4,5 are labelled to be 1 and 1,2 are labelled as 0. Discard the reviews with rating 3'
# In[49]:
# In[50]:
pos_label = df[df["star_rating"].isin([4,5])]
pos_label["label"]=1
# In[51]:
neutral_label = df[df["star_rating"].isin([3])]
# In[52]:
neg_label = df[df["star_rating"].isin([1,2])]
neg_label["label"]=0
class_stats=[pos_label.shape[0],neutral_label.shape[0],neg_label.shape[0]]
print(*class_stats,sep=",")
#  ## We select 200000 reviews randomly with 100,000 positive and 100,000 negative reviews.
# 
# 
# In[53]:
rnd_pos_labels = pos_label.sample(n=100000)
rnd_neg_labels = neg_label.sample(n=100000)
# In[54]:
exp_data = pd.concat([rnd_pos_labels,rnd_neg_labels]).reset_index(drop=True)
# # Data Cleaning
# 
# ## Convert the all reviews into the lower case.
# In[55]:
## Stats Data Frame stores the count of characters after each pre-processing task
stats = pd.DataFrame()
stats['count_before_clean'] = exp_data['review_body'].str.len()
# In[56]:
exp_data['review_body'] = exp_data['review_body'].str.lower()
# ## Remove the HTML and URLs from the reviews
# In[58]:
exp_data['after_url_clean'] = exp_data['review_body'].apply(lambda x: bs4.BeautifulSoup(x, 'lxml').get_text())
exp_data['after_url_clean'] = exp_data['after_url_clean'].apply(lambda x: re.sub(r'http\S+', '', x))
stats['count_after_url_clean'] = exp_data['after_url_clean'].str.len()
# ## Strip Whitespaces
# In[59]:
exp_data['after_space_clean'] = exp_data['after_url_clean'].str.strip()
stats['count_after_spaces_clean'] = exp_data['after_space_clean'].str.len()
# ## Perform contractions on the reviews.
# In[60]:
exp_data['after_contraction_fix'] = exp_data['after_space_clean'].apply(lambda x: [contractions.fix(word) for word in x.split()])
exp_data['after_contraction_fix'] = [' '.join(map(str, l)) for l in exp_data['after_contraction_fix']]
stats['count_after_expanding_contractions'] = exp_data['after_contraction_fix'].str.len()
# ## Remove non-alphabetical characters
# In[61]:
exp_data['after_nonalpha_clean'] = exp_data.after_contraction_fix.str.replace(r'[^a-zA-Z]\s?',r' ',regex=True)
stats['count_after_nonalpha_clean'] = exp_data['after_nonalpha_clean'].str.len()
# ## Remove the extra spaces between the words
# In[62]:
exp_data['after_nonalpha_clean'] = exp_data['after_nonalpha_clean'].str.strip()
stats['count_after_nonalpha_clean'] = exp_data['after_nonalpha_clean'].str.len()
cleaning_stats = [stats['count_before_clean'].mean(),stats['count_after_nonalpha_clean'].mean()]
print(*cleaning_stats, sep=",")
# # Pre-processing
# ## Remove the stop words 
# In[63]:
from nltk.corpus import stopwords
stop = stopwords.words('english')
exp_data['after_stopwords_removal'] = exp_data['after_nonalpha_clean'].apply(lambda x: ' '.join([x for x in x.split() if x not in stop]))
stats['count_after_removing_stopwords'] = exp_data['after_stopwords_removal'].str.len()
# ## Perform Lemmatization  
# In[64]:
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
exp_data['after_lemmatization'] =  exp_data['after_stopwords_removal'].apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x)]))
stats['count_after_lemmatization'] = exp_data['after_lemmatization'].str.len()
prepro_stats = [stats['count_after_nonalpha_clean'].mean(),stats['count_after_lemmatization'].mean()]
print(*prepro_stats,sep=",")
# # TF-IDF Feature Extraction
# In[125]:
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(exp_data['after_lemmatization'], exp_data['label'], test_size=0.2, random_state=30)
# In[126]:
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer= TfidfVectorizer()
tf_x_train = vectorizer.fit_transform(X_train)
tf_x_test = vectorizer.transform(X_test)
# # Perceptron
# In[127]:
from sklearn.linear_model import Perceptron
perceptron = Perceptron(tol=1e-3, random_state=0)
perceptron.fit(tf_x_train,Y_train)
y_test_pred=perceptron.predict(tf_x_test)
y_train_pred=perceptron.predict(tf_x_train)
test_report=classification_report(Y_test,y_test_pred,output_dict=True)
train_report=classification_report(Y_train,y_train_pred,output_dict=True)
train_metrics = [train_report['accuracy'],train_report['1']['precision'],train_report['1']['recall'],train_report['1']['f1-score']]
print(*train_metrics,sep="\n")
test_metrics = [test_report['accuracy'],test_report['1']['precision'],test_report['1']['recall'],test_report['1']['f1-score']]
print(*test_metrics,sep="\n")
# # SVM
# In[130]:
from sklearn.svm import LinearSVC
svm = LinearSVC(random_state=0)
svm.fit(tf_x_train,Y_train)
y_test_pred=svm.predict(tf_x_test)
y_train_pred=svm.predict(tf_x_train)
test_report=classification_report(Y_test,y_test_pred,output_dict=True)
train_report=classification_report(Y_train,y_train_pred,output_dict=True)
train_metrics = [train_report['accuracy'],train_report['1']['precision'],train_report['1']['recall'],train_report['1']['f1-score']]
print(*train_metrics,sep="\n")
test_metrics = [test_report['accuracy'],test_report['1']['precision'],test_report['1']['recall'],test_report['1']['f1-score']]
print(*test_metrics,sep="\n")
# # Logistic Regression
# In[133]:
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000,solver='saga')
lr.fit(tf_x_train,Y_train)
y_test_pred=lr.predict(tf_x_test)
y_train_pred=lr.predict(tf_x_train)
test_report=classification_report(Y_test,y_test_pred,output_dict=True)
train_report=classification_report(Y_train,y_train_pred,output_dict=True)
train_metrics = [train_report['accuracy'],train_report['1']['precision'],train_report['1']['recall'],train_report['1']['f1-score']]
print(*train_metrics,sep="\n")
test_metrics = [test_report['accuracy'],test_report['1']['precision'],test_report['1']['recall'],test_report['1']['f1-score']]
print(*test_metrics,sep="\n")
# # Naive Bayes
# In[138]:
from sklearn.naive_bayes import MultinomialNB
mul_model = MultinomialNB()
mul_model.fit(tf_x_train,Y_train)
y_train_pred = mul_model.predict(tf_x_train)
y_test_pred = mul_model.predict(tf_x_test)
test_report=classification_report(Y_test, y_test_pred,output_dict=True)
train_report=classification_report(Y_train, y_train_pred,output_dict=True)
train_metrics = [train_report['accuracy'],train_report['1']['precision'],train_report['1']['recall'],train_report['1']['f1-score']]
print(*train_metrics,sep="\n")
test_metrics = [test_report['accuracy'],test_report['1']['precision'],test_report['1']['recall'],test_report['1']['f1-score']]
print(*test_metrics,sep="\n")