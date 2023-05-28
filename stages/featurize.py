import numpy as np
import pandas as pd
import nltk
import nltk.corpus
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize.toktok import ToktokTokenizer
from wordcloud import WordCloud
import string
import os
import yaml

def featurize() -> None:
    os.chdir('C:/Users/Ayo Agbaje/Documents/Code/Python/GIGS/PYTHON_docs/py_files/Sentiment_Analysis')
    with open('params.yaml') as config_:
        config__ = yaml.safe_load(config_)

    df_ = pd.read_csv(config__['data']['processed']['data_7'])

    # Convert all text to lowercase
    def lower_case(text_):
        return text_.lower()

    df_['Comments'] = df_['Comments'].apply(lower_case)

    # Remove stopwords
    def remove_stopwords(text_):
        wst = WhitespaceTokenizer()
        tokens_ = wst.tokenize(text_)
        stop_words_ = stopwords.words('english')
        stop_words_.remove('not')
        stop_words_.remove('no')

        stopword_free_text_ = []
        for words_ in tokens_:
            if words_ not in stop_words_:
                stopword_free_text_.append(words_)
        
        new_text_ = ' '.join(stopword_free_text_)
        return new_text_


    df_['Comments'] = df_['Comments'].apply(remove_stopwords)

    # Remove Punctuations
    def remove_punctuations(text_):
        # ttt = ToktokTokenizer()
        tokens_ = text_
        punctuations_ = string.punctuation

        punctuation_free_text_ = []
        for i in tokens_:
            if i not in punctuations_:
                punctuation_free_text_.append(i)
        
        new_text_ = ''.join(punctuation_free_text_)
        return new_text_


    df_['Comments'] = df_['Comments'].apply(remove_punctuations)


    # Remove stopwords again
    def remove_stopwords(text_):
        ttt = ToktokTokenizer()
        tokens_ = ttt.tokenize(text_)
        stop_words_ = stopwords.words('english')
        stop_words_.remove('not')
        stop_words_.remove('no')

        stopword_free_text_ = []
        for words_ in tokens_:
            if words_ not in stop_words_:
                stopword_free_text_.append(words_)
        
        new_text_ = ' '.join(stopword_free_text_)
        return new_text_
    
    # Remove all hashtags and user tags
    def remove_tags(text_):
        wst = WhitespaceTokenizer()
        tokens_ = wst.tokenize(text_)

        tag_free_text_ = []
        for word_ in tokens_:
            if '@' not in word_ or '#' not in word_:
                tag_free_text_.append(word_)
        
        text__ = ' '.join(tag_free_text_)
        return text_
    
    df_['Comments'] = df_['Comments'].apply(remove_tags)

    # Remove all New Lines
    def remove_new_lines(text_):
        return text_.replace('\n', '')
    

    df_['Comments'] = df_['Comments'].apply(remove_new_lines)

    df_.to_csv(config__['paths']['processed']['data_2'], index = 0)

    print('Features have been extracted and all new datasets saved to disk')

if __name__ == '__main__':
        featurize()