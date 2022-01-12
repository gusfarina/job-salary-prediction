import re
import string
import pickle
import pandas as pd
import numpy as np
from numpy import absolute

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import xgboost as xgb
from xgboost import XGBRegressor

import nltk
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except FileExistsError:
    pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class SalaryPredictor:
    def __init__(self):
        self.path_to_training_data = None
        self.X_column_names = None
        self.y_column_name = None
        self.word_vectorizer = None
        self.X = None
        self.y = None
        self.kfold_mae_score = None
        self.fitted_model = None

    # def __text_sanitization(self, content):
    #     # Making its content lower case
    #     content = content.lower()

    #     # Removing HTML Tags
    #     html_removal_code = re.compile('<.*?>') 
    #     content = re.sub(html_removal_code, '', content)

    #     # Removing ponctuation
    #     content = content.translate(str.maketrans("", "", string.punctuation))

    #     # Removing white spaces
    #     content = content.strip()

    #     return content

    def __text_sanitization(self, content):
        stop_words = set(stopwords.words("english"))
        
        # Making its content lower case
        content = content.lower()

        # Removing HTML Tags
        html_removal_code = re.compile('<.*?>') 
        content = re.sub(html_removal_code, '', content)

        # Removing ponctuation
        content = content.translate(str.maketrans("", "", string.punctuation))

        # Removing white spaces
        content = content.strip()
        
        # Removing stop words
        word_tokens = word_tokenize(content)
        filtered_text = ''
        for word in word_tokens:
            if word not in stop_words:
                filtered_text = filtered_text + word + " "
        content = filtered_text.strip()

        return content


    def __data_preprocessing(self):
        print('Processing Data...')
        train = pd.read_csv(self.path_to_training_data)

        X = train[self.X_column_names].fillna(0)
        self.y = train[self.y_column_name].fillna(0)

        X['concat_features'] = X.astype(str).apply(' '.join, axis=1)
        X.concat_features = X.concat_features.apply(self.__text_sanitization)

        self.X = X.concat_features
        
        # self.word_vectorizer = TfidfVectorizer(
        #     sublinear_tf=True,
        #     strip_accents='unicode',
        #     analyzer='word',
        #     token_pattern=r'\w{1,}',
        #     ngram_range=(1, 1),
        #     norm='l2',
        #     min_df=0,
        #     smooth_idf=False,
        #     max_features=15000)

        self.word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(2, 2),
            norm='l2',
            min_df=0,
            smooth_idf=False,
            max_features=7000)

        self.word_vectorizer.fit(self.X)

        with open('word_vec', 'wb') as file:
            pickle.dump(self.word_vectorizer, file)


    def __model_creation(self):
        print('Creating model...')
        X_vec = self.word_vectorizer.transform(self.X)

        xgbr = XGBRegressor(
            random_state=12, 
            n_estimators=150,
            max_depth=8, 
            learning_rate=0.4,
            verbosity=1
        )

        cv = KFold(n_splits=5)
        scores = cross_val_score(xgbr, X_vec, self.y, scoring='neg_mean_absolute_error', cv=cv)
        scores = absolute(scores)

        self.kfold_mae_score = scores.mean()

        self.fitted_model = xgbr.fit(X_vec, self.y)

        with open('model', 'wb') as file:
            pickle.dump(self.fitted_model, file)


    def train(self, path_to_training_data, X_column_names, y_column_name):
        print('Begining Training Process!')

        self.path_to_training_data = path_to_training_data
        self.X_column_names = X_column_names
        self.y_column_name = y_column_name

        self.__data_preprocessing()
        self.__model_creation()

        print('End of Training Process!')
        print(f"Kfold MAE Model Score: {self.kfold_mae_score}")


    def predict(self, X):
        try:
            with open('model', 'rb') as file:
                self.fitted_model = pickle.load(file)

            with open('word_vec', 'rb') as file:
                self.word_vectorizer = pickle.load(file) 
        except BaseException as err:
            print("Something went wrong wile retrieving the model and the word vectorizer \n")
            print(f"Error: {err}")

        concat_X = ' '.join(X)
        concat_X = self.__text_sanitization(concat_X)

        X_vec = self.word_vectorizer.transform([concat_X])
        return self.fitted_model.predict(X_vec) 


if __name__ == '__main__':
    pass