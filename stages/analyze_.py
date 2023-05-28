import numpy as np
import pandas as pd
import nltk
import nltk.corpus
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import PIL
from PIL import ImageFont
from wordcloud import WordCloud
plt.style.use('fivethirtyeight')
import os
import yaml


def analyze() -> None:
    os.chdir('C:/Users/Ayo Agbaje/Documents/Code/Python/GIGS/PYTHON_docs/py_files/Sentiment_Analysis')
    with open('params.yaml') as config_:
        config__ = yaml.safe_load(config_)

    df_ = pd.read_csv(config__['data']['processed']['data_1'])
    sentiment_map = {
        1: 'Racist',
        0: 'Not Racist'
    }   

    df_['Classification'] = df_['Classification'].map(sentiment_map)
    sentiment_count_ = df_['Classification'].value_counts()
    sentiment_count_ = pd.DataFrame(data = sentiment_count_)
    sentiment_count_.columns = ["COUNT"]

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7,7), facecolor = 'white')
    sentiment_count_['COUNT'].plot.pie(ax = ax, textprops = {'color': 'black', 'weight': 'bold', 'size': 15}, ylabel = '', autopct = '%.1f',
                                   colors = ['red', 'blue', 'grey'], shadow = False, explode = (0.06, 0.06), startangle = 90,
                                   wedgeprops = {'linewidth': 2, 'edgecolor': 'black'})
    plt.title('PIE CHART SHOWING DISTRIBUTION OF SENTIMENT OF TWEETS', fontweight = 'bold')
    plt.savefig(config__['paths']['plots']['img_1'])

    # All Racist comments wordcloud
    stopwords_ = stopwords.words('english')
    racist_df_ = df_[df_['Classification'] == 'Racist']
    racist_comments_ = []
    for i in racist_df_['Comments'].values:
        racist_comments_.append(i)

    racist_comments_ = ' '.join(racist_comments_)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (11,11))

    word_cloud = WordCloud(max_words = 450, width = 500, height = 500, stopwords = stopwords_)
    img_ = word_cloud.generate(racist_comments_)
    plt.imshow(img_)
    # plt.imsave(fname = config__['paths']['plots']['img_1'], arr = img_)
    img_.to_file(config__['paths']['plots']['img_2'])

    # All non-Racist comments wordcloud
    stopwords_ = stopwords.words('english')
    non_racist_df_ = df_[df_['Classification'] == 'Not Racist']
    non_racist_comments_ = []
    for i in non_racist_df_['Comments'].values:
        non_racist_comments_.append(i)

    non_racist_comments_ = ' '.join(non_racist_comments_)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (11,11))

    word_cloud = WordCloud(max_words = 450, width = 500, height = 500, stopwords=stopwords_)
    img_ = word_cloud.generate(non_racist_comments_)
    plt.imshow(img_)
    # plt.imsave(fname = config__['paths']['plots']['img_3'], arr = img_)
    img_.to_file(config__['paths']['plots']['img_3'])

    sentiment_map = {
        'Racist': 1,
        'Not Racist': 0
    }

    df_['Classification'] = df_['Classification'].map(sentiment_map)
    df_.to_csv(config__['paths']['processed']['data_7'], index = 0)


    print('All Data Analyzed and plots saved')

if __name__ == '__main__':
    analyze()