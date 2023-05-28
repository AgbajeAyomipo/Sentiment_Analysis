import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, precision_score
import yaml
import os
import pickle


def train() -> None:
    os.chdir('C:/Users/Ayo Agbaje/Documents/Code/Python/GIGS/PYTHON_docs/py_files/Sentiment_Analysis')
    with open('params.yaml') as config_:
        config__ = yaml.safe_load(config_)

    xtrain = pd.read_csv(config__['data']['processed']['data_3'])
    xtest = pd.read_csv(config__['data']['processed']['data_4'])
    ytrain = pd.read_csv(config__['data']['processed']['data_5'])
    # ytest = pd.read_csv(config__['data']['processed']['data_6'])

    x_train = xtrain['Comments']
    y_train = ytrain['Classification']
    x_test = xtest['Comments']
    # y_test = ytest['Classification']

    bow = CountVectorizer(
            min_df = config__['bow']['min_df'],
            max_df = config__['bow']['max_df'],
            binary = False,
            ngram_range=(1,3)
            )

    bow.fit(x_train)
    cv_x_train = bow.transform(x_train)


    alg_pick = config__['train']['alg']
    if alg_pick == 'mnb':
        alg_ = MultinomialNB()
    elif alg_pick == 'lr':
        alg_ = LogisticRegression(solver='lbfgs', max_iter=100)
    elif alg_pick == 'svc':
        alg_ = SVC()

    alg_.fit(cv_x_train, y_train)

    model_path = open(config__['paths']['models']['one_'], 'wb')

    pickle.dump(
        obj = alg_,
        file = model_path
    )

    print('Model Successfully Trained')

if __name__ == '__main__':
    train()

