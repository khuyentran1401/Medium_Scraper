import datapane as dp
import csv
import pandas as pd
import numpy as np
import re
import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import csr_matrix, hstack, csc_matrix
from mlflow import log_metric, log_param, log_artifact

import joblib

import logging

import datapane as dp 






class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        
        self.variables = variables

    def fit(self, X, y=None):

        return self 

    def transform(self, X):

        X = X.copy()

        for feature in self.variables:

            X[feature] = X[feature].fillna('None')

        return X


class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):

        self.variables = variables

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X = X.copy()

        for feature in self.variables:

            X[feature] = X[feature].fillna(0)

        return X

class TitleImputer(BaseEstimator, TransformerMixin):
    '''Extract missing tittle from url'''

    def __init__(self):

        pass

    def fit(self, X, y=None):

        return self 

    def url_to_title(self, url):
        '''Find title from url'''

        url = url.replace('https://towardsdatascience.com/', '')
        url = url.replace('https://medium.com/', '')
        url = re.sub(r'.*/', '', url)
        url = re.sub(r'([A-Za-z]+[\d@]+[\w@]*|[\d@]+[A-Za-z]+[\w@]*).+', '', url)
        title = url.replace('-', ' ')

        return title

    def transform(self, X):

        if X.url.isnull().any():
            return X

        X = X.copy()

        null_urls = (list(X.loc[X.Title.isnull(), 'url']))

        null_titles = []

        for url in null_urls:
            null_titles.append(self.url_to_title(url))

        X.loc[X.Title.isnull(), 'Title'] = null_titles

        return X 


class ClapsToNumerical(BaseEstimator, TransformerMixin):
    '''Transform claps to numerical data'''

    def __init__(self):

        pass 

    def fit(self, X, y=None):


        return self

    def str_to_float(self, feature):
        '''Change string with K or M to a float (.i.e, 5k)'''

        feature = feature.replace(r'[KM]+$', '', regex=True).astype(float) * \
            feature.str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(
                1).replace(['K', 'M'], [10**3, 10**6]).astype(int)

        return feature
    
    def transform(self, X):

        X = X.copy()
        X.Claps = self.str_to_float(X.Claps)

        cut_bins = [-1, 10, 100, 1000, 10000, 26000]
        X['Claps'] = pd.cut(X['Claps'], bins=cut_bins)

        X.Claps = X.Claps.cat.rename_categories([0, 1, 2, 3, 4])

        return X.Claps


class AddFrequency(BaseEstimator, TransformerMixin):
    '''Add frequency of the data'''

    def __init__(self, frequent_variables=None):

        self.frequent_variables = frequent_variables 

    def fit(self, X, y=None):
        return self 

    def transform(self, X):

        X = X.copy()

        for feature in self.frequent_variables:

            X[feature + '_count'] = X.groupby([feature])['Title'].transform('count')

        return X


class FindWeekDay(BaseEstimator, TransformerMixin):
    '''Find the day of the week'''

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self 

    def date_to_weekday(self,year, month, day):
        '''Find the day of the week with regarding to the date'''

        return datetime.date(year, month, day).weekday()

    def transform(self, X):

        years = list(X.Year)
        months = list(X.Month)
        days = list(X.Day)

        week_days = []

        for year, month, day in zip(years, months, days):

            week_days.append(self.date_to_weekday(year, month, day))

        X['week_days'] = week_days

        return X 


class DropFeatures(BaseEstimator, TransformerMixin):
    '''Drop unnecessary columns'''

    def __init__(self, variables: list):

        self.variables = variables

    def fit(self, X, y=None):

        return self 

    def transform(self, X):

        X.copy()
        X = X.drop(self.variables, axis=1)

        return X 


class CategoricalToNumerical(BaseEstimator, TransformerMixin):
    '''Turn categorical data to numerical'''

    def __init__(self):

        pass

    def fit(self, X, y=None):
        
        frequencies = X.Publication.value_counts(
            normalize=True, ascending=True)

        self.threshold = frequencies[(frequencies.cumsum() > 0.2).idxmax()]

        return self 

    def transform(self, X):

        X = X.copy()
        
        X.Publication = X['Publication'].mask(X['Publication'].map(
            X['Publication'].value_counts(normalize=True)) < self.threshold, 'Other')

        X = pd.get_dummies(X, columns=['Publication', 'Tag'])

        return X 

class ProcessMatrix(BaseEstimator, TransformerMixin):
    '''Vectorize text features and transform the df to sparse matrix'''

    def __init__(self):
        pass

    def fit(self, X, y=None):

        return self 

    def transform(self, X):
        X = X.copy()

        X = hstack((csc_matrix(X), csc_matrix(np.ones((X.shape[0], 1)))))


        return X


class ProcessText(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        
        self.variables=variables

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X=X.copy()

        for feature in self.variables:

            processed = []

            for text in X[feature]:

                # lowercase
                text = text.lower()

                #remove punctuation
                text = text.translate(
                    str.maketrans('', '', string.punctuation))

                #remove stopwords
                stop_words = set(stopwords.words('english'))

                #tokenize
                tokens = word_tokenize(text)
                new_text = [i for i in tokens if not i in stop_words]

                new_text = ' '.join(new_text)

                processed.append(new_text)


        return processed


        










 


        








    

    


    
    





