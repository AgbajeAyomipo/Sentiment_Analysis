import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split, KFold
import yaml
import os


def data_split() -> None:
    os.chdir('C:/Users/Ayo Agbaje/Documents/Code/Python/GIGS/PYTHON_docs/py_files/Sentiment_Analysis')
    with open('params.yaml') as config_:
        config__ = yaml.safe_load(config_)

    df_ = pd.read_csv(config__['data']['processed']['data_2'])
    df_ = df_.dropna(axis = 0)
    
    X = df_['Comments']
    y = df_['Classification']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=config__['split_data']['train_size'],
                                                        random_state=config__['split_data']['random_state'])
    # X_train.shape, X_test.shape

    xtrain_df = pd.DataFrame(data = X_train, columns = ['Comments'])
    xtest_df = pd.DataFrame(data = X_test, columns = ['Comments'])
    ytrain_df = pd.DataFrame(data = y_train, columns = ['Classification'])
    ytest_df = pd.DataFrame(data = y_test, columns = ['Classification'])

    xtrain_df.to_csv(config__['paths']['processed']['data_3'], index = 0)
    xtest_df.to_csv(config__['paths']['processed']['data_4'], index = 0)
    ytrain_df.to_csv(config__['paths']['processed']['data_5'], index = 0)
    ytest_df.to_csv(config__['paths']['processed']['data_6'], index = 0)

    print('All Data Splitted and saved to disk')

if __name__ == '__main__':
    data_split()