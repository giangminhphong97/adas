import os
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read data
# Use the correct dataset folder (it lives at `fakenews/dataset/news.csv`)
data_dir = 'fakenews/dataset'
data_path = os.path.join(data_dir, 'news.csv')
if not os.path.exists(data_path):
	raise FileNotFoundError(f"Dataset not found at {data_path}. Expected file at fakenews/dataset/news.csv")
df = pd.read_csv(data_path)

#Get shape and head
df.shape
df.head()

# Get the labels
labels = df.label
labels.head()

# Split the datase
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

# Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
#print(f'Accuracy: {round(score*100,2)}%')

# Build confusion matrix
[[true_pos,false_neg],[false_pos,true_neg]] = confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])
print(true_pos)
print(false_neg)
print(false_pos)
print(true_neg)