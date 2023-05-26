import numpy as np
import pandas as pd
import yaml
import os


def data_load() -> None:
    os.chdir('C:/Users/Ayo Agbaje/Documents/Code/Python/GIGS/PYTHON_docs/py_files/Sentiment_Analysis')
    with open('params.yaml') as config_:
        config__ = yaml.safe_load(config_)

    df_ = pd.read_csv(config__['data']['raw']['data_1'])
    df_.columns = ['id', 'Classification', 'Comments']
    df_ = df_.dropna(axis = 0)

    df_.to_csv(config__['paths']['processed']['data_1'], index = 0)
    print('Data Loaded Successfully')


if __name__ == '__main__':
    data_load()