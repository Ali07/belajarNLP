import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier as RFC

train = pd.read_csv("nlp_data/labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords

def review_to_words( raw_review ):
    #remove html tag
    review_text = bs(raw_review,"html5lib").get_text()
    #remove non letter
    letters_only  = re.sub("[^a-zA-Z]", " ", review_text)
    #convert lower case
    words = letters_only.lower().split()
    #stopword
    stops = set(stopwords.words("english"))
    #remove stop words
    meaningful_words = [w for w in words if not w in stops]
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))

#get size of review
num_reviews = train["review"].size

#buat variable
print("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []
for i in range( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print("Review %d of %d\n" % ( i+1, num_reviews ))                                                                    
    clean_train_reviews.append( review_to_words( train["review"][i] ))
    
print("Creating the bag of words...\n")

#membuat bag of words

#bag of words tools
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

train_data_features = vectorizer.fit_transform(clean_train_reviews)

#convert result to array
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()
print("Training the random forest...")
#inisialisasi random forest
forest = RFC(n_estimators = 100)

#memasukkan data kedalam random forest
forest = forest.fit(train_data_features, train["sentiment"] )
    