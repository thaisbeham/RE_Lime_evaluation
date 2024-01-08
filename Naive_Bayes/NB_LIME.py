from __future__ import print_function
import pandas as pd
import nltk
from nltk.corpus import stopwords
#nltk.download("stopwords") 
from tqdm import tqdm
import csv

#from utils import read_docs_from_csv, split_train_dev_test, classif_metrics

import lime
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics

from sklearn.naive_bayes import GaussianNB
from lime import lime_text
from sklearn.pipeline import make_pipeline

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
#from lime.lime_text import LimeTextExplainer
from extended_lime_explainer import ExtendedLimeTextExplainer

def read_json(file_path):
    df = pd.read_json(file_path)
    df = df[['id', 'relation', 'token', 'subj_start', 'obj_start']]

    df['sentence'] = [' '.join(text) for text in  df['token']]
    df['subj_word'] = df.apply(lambda row: row['token'][row['subj_start']] if row['subj_start'] < len(row['token']) else None, axis=1)
    df['obj_word'] = df.apply(lambda row: row['token'][row['obj_start']] if row['obj_start'] < len(row['token']) else None, axis=1)

    return df

df_train = read_json('../Data/train.json')
df_test = read_json('../Data/test.json')

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer(stop_words=None)
X_train_vectorized = vectorizer.fit_transform(df_train.sentence)
X_test_vectorized = vectorizer.transform(df_test.sentence)

# Create and train the Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectorized, df_train.relation)

# Make predictions on the test set
y_pred = clf.predict(X_test_vectorized)

y_test = df_test.relation
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

classification_report = classification_report(y_test, y_pred)

class_labels = clf.classes_

c = make_pipeline(vectorizer, clf)

explainer = ExtendedLimeTextExplainer(class_names=class_labels)

def explainer_c():

    c = make_pipeline(vectorizer, clf)
    explainer = ExtendedLimeTextExplainer(class_names=class_labels)

    return explainer, c 

