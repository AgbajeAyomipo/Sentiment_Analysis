import numpy as np
import pandas as pd
import yaml
import os
from cleantext import clean

def data_load() -> None:
    os.chdir('C:/Users/Ayo Agbaje/Documents/Code/Python/GIGS/PYTHON_docs/py_files/Sentiment_Analysis')
    with open('params.yaml') as config_:
        config__ = yaml.safe_load(config_)

    df_ = pd.read_csv(config__['data']['raw']['data_1'])
    df_.columns = ['id', 'Classification', 'Comments']
    df_ = df_.dropna(axis = 0)

    # clean_text
    def clean_text(text_):
        return clean(text_, no_emoji = True)

    df_['Comments'] = df_['Comments'].apply(clean_text)

    df_.to_csv(config__['paths']['processed']['data_1'], index = 0)
    print('Data Loaded Successfully')


if __name__ == '__main__':
    data_load()