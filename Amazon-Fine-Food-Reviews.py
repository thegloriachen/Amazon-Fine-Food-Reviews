"""
File: game of thrones_decision tree.py
Name: 513716004 陳映廷(Gloria)
---------------------------
This file shows how to use pandas and sklearn
packages to build a model within random forest
and evaluate the accuracy within k-fold cross-validation.
"""

import pandas as pd
from sklearn import ensemble
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import gensim
import numpy as np
import re
import string


# Constants - filenames for data set
TRAIN_FILE = 'sentiment analysis/Reviews.csv'


def main():
    # Data preprocessing
    data = data_preprocess(TRAIN_FILE)
    # print(data.head(30))

    # Inspecting if there is any NaN data
    # print(data.isna().sum())

    # Extract true labels
    y = data.Score
    # print("This is true labels", y)

    # Extract features
    x = data.Text
    # print("This is features", x)

    # Construct the model and evaluate the accuracy by using tf-idf to process the data.

    # Using CountVectorizer and TfidfTransformer to transform the data into vectors.
    vectorizer = CountVectorizer()
    x_counts = vectorizer.fit_transform(x)
    tfidf_transformer = TfidfTransformer()
    x_tfidf = tfidf_transformer.fit_transform(x_counts)

    # Inspect x_tfidf shape
    # print(f"x_tfidf shape: {x_tfidf.shape}")
    # print(vectorizer.get_feature_names_out())

    # Splitting the data into training data and testing data (75% Training data, 25% Testing data)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_tfidf, y, train_size=0.75)

    # Construct Forest by the result of text data witch using TfidfTransformer to process it.
    forest = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
    forest_classifier = forest.fit(x_train, y_train)

    # Make predictions by random forest classifier by the result of text data
    # which using TfidfTransformer to process it.
    predictions = forest_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Random Forest Model Accuracy(TfidfTransformer): {accuracy}")

    # To analyze the score within k-fold cross-validation by the result of text data
    # which using TfidfTransformer to process it.
    # k-fold cross-validation, k=4
    cv_scores = cross_val_score(forest_classifier, x_tfidf, y, cv=4, scoring='accuracy')
    print(f"Cross-validation scores for each fold(TfidfTransformer): {cv_scores}")
    print(f"Average cross-validation accuracy(TfidfTransformer): {np.mean(cv_scores)}")

    # Construct the model and evaluate the accuracy by using Word2vec to process the data.

    # Using Word2vec to transform the data into vectors.
    d_text = data.Text.apply(lambda x: x.split())
    model = gensim.models.Word2Vec(sentences=d_text, vector_size=100, window=5, min_count=1, workers=4)

    # Inspect the data
    # print(model.wv.key_to_index)

    # To get average of Word2vec for every sentence.
    def get_average_word2vec(sentence, model):
        words = [word for word in sentence if word in model.wv]
        if len(words) > 0:
            return np.mean([model.wv[word] for word in words], axis=0)
        else:
            # If all the words in a sentence not in Word2Vec model, return 0.
            return np.zeros(model.vector_size)
    x_word2vec = np.array([get_average_word2vec(sentence, model) for sentence in d_text])

    # Inspect Word2vec shape
    # print(f"x_word2vec shape: {x_word2vec.shape}")

    # Splitting the data into training data and testing data (75% Training data, 25% Testing data)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_word2vec, y, train_size=0.75)

    # Construct Forest by the result of text data witch using Word2vec to process it.
    forest = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
    forest_classifier = forest.fit(x_train, y_train)

    # Make predictions by random forest classifier by the result of text data
    # which using TfidfTransformer to process it.
    predictions = forest_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Random Forest Model Accuracy(Word2vec): {accuracy}")

    # To analyze the score within k-fold cross-validation by the result of text data
    # which using TfidfTransformer to process it.
    # k-fold cross-validation, k=4
    cv_scores = cross_val_score(forest_classifier, x_word2vec, y, cv=4, scoring='accuracy')
    print(f"Cross-validation scores for each fold(Word2vec): {cv_scores}")
    print(f"Average cross-validation accuracy(Word2vec): {np.mean(cv_scores)}")


def data_preprocess(filename):
    """
    : param filename: str, the csv file to be read into by pd
    """
    # Read in data as a column based DataFrame
    data = pd.read_csv(filename, nrows=10000, usecols=['Text', 'Score'])

    # If Score<4, replacing the data with 0 --> means negative.
    # If Score>=4, replacing the data with 1 --> means positive.
    data.loc[data.Score < 4, 'Score'] = 0
    data.loc[data.Score >= 4, 'Score'] = 1

    # Data cleaning
    # Change the texts to lowercase letters, and remove HTML tags, punctuation, number.
    html_tags = re.compile('<.*?>')
    d_text = data['Text']
    d_text = d_text.str.lower()
    d_text = d_text.str.replace(html_tags, ' ', regex=True)
    d_text = d_text.str.replace('[-|?|!|\'|"|#|.|,|)|(|\|/|*|>|<|&|$|_|º|¼|½|î]', ' ', regex=True)
    d_text = d_text.str.replace('[0123456789]', ' ', regex=True)

    # Eliminate stop words
    d_text_transformed_list = d_text.apply(lambda x: ' '.join([word for word in x.split() if word.lower()
                                                               not in ENGLISH_STOP_WORDS and word.lower()
                                                               not in list(string.ascii_lowercase)]))

    # # Split the data
    # d_text_transformed_list = d_text_transformed_list.apply(lambda x: x.split())

    # Assign the transformed data to dataframe
    data['Text'] = d_text_transformed_list

    return data


if __name__ == '__main__':
    main()
