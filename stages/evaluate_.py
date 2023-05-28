import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, precision_score
import yaml
import os
import pickle
import json

def evaluate() -> None:
    os.chdir('C:/Users/Ayo Agbaje/Documents/Code/Python/GIGS/PYTHON_docs/py_files/Sentiment_Analysis')
    with open('params.yaml') as config_:
        config__ = yaml.safe_load(config_)

    xtrain = pd.read_csv(config__['data']['processed']['data_3'])
    xtest = pd.read_csv(config__['data']['processed']['data_4'])
    ytrain = pd.read_csv(config__['data']['processed']['data_5'])
    ytest = pd.read_csv(config__['data']['processed']['data_6'])

    x_train = xtrain['Comments']
    y_train = ytrain['Classification']
    x_test = xtest['Comments']
    y_test = ytest['Classification']

    bow = CountVectorizer(
            min_df = config__['bow']['min_df'],
            max_df = config__['bow']['max_df'],
            binary = False,
            ngram_range=(1,3)
            )

    bow.fit(x_train)
    cv_x_train = bow.transform(x_train)
    cv_x_test = bow.transform(x_test)

    model_path = open(config__['paths']['models']['one_'], 'rb')
    model = pickle.load(model_path)
    preds_ = model.predict(cv_x_test)

    accuracy_score_ = accuracy_score(y_true = y_test, y_pred = preds_)
    precision_score_ = precision_score(y_true = y_test, y_pred = preds_)

    metric_ = {
        'accuracy_score': accuracy_score_,
        'precision_score': precision_score_
    }

    metric_path = open(config__['paths']['metrics']['two_'], 'w')
    json.dump(
        obj = metric_,
        fp = metric_path,
        sort_keys=False,
        indent = 4
    )

    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8,8), facecolor = 'white')
    plot_ = ConfusionMatrixDisplay.from_predictions(y_true = y_test, y_pred = preds_, ax = ax,
                                                    cmap = 'Blues', colorbar = False, display_labels=['Not Racist', 'Racist'])
    ax.set_title('CONFUSION MATRIX')
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14, rotation = 45)

    plt.savefig(config__['paths']['plots']['img_4'])

    print('All necessary evaluations done, Results and plots saved')


if __name__ == '__main__':
    evaluate()

