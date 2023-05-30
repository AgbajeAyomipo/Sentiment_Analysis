import gradio as gr
import nltk
import nltk.corpus
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize.toktok import ToktokTokenizer
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import string
import numpy as np
import pandas as pd
from cleantext import clean


def test_for_racism(text_):
    text_ = text_.lower()
    text_ = clean(text_, no_emoji = True)

    wst = WhitespaceTokenizer()
    tokens_ = wst.tokenize(text_)
    stop_words_ = stopwords.words('english')
    stop_words_.remove('not')
    stop_words_.remove('no')

    stopword_free_text_ = []
    for words_ in tokens_:
        if words_ not in stop_words_:
            stopword_free_text_.append(words_)
    
    text_ = ' '.join(stopword_free_text_)

    tokens_ = text_
    punctuations_ = string.punctuation

    punctuation_free_text_ = []
    for i in tokens_:
        if i not in punctuations_:
            punctuation_free_text_.append(i)
    
    text_ = ''.join(punctuation_free_text_)

    ttt = ToktokTokenizer()
    tokens_ = ttt.tokenize(text_)
    stop_words_ = stopwords.words('english')
    stop_words_.remove('not')
    stop_words_.remove('no')

    stopword_free_text_ = []
    for words_ in tokens_:
        if words_ not in stop_words_:
            stopword_free_text_.append(words_)
    
    text_ = ' '.join(stopword_free_text_)

    wst = WhitespaceTokenizer()
    tokens_ = wst.tokenize(text_)

    tag_free_text_ = []
    for word_ in tokens_:
        if '@' not in word_ or '#' not in word_:
            tag_free_text_.append(word_)
    
    text_ = ' '.join(tag_free_text_)
    # text_ = text_.replace('\n', '')

    array_ = [text_]
    data = {
        'comment': array_
    }
    df_ = pd.DataFrame(data = data)

    x = df_['comment']
    
    xtrain = pd.read_csv('data/xtrain.csv')
    xtest = pd.read_csv('data/xtest.csv')
    ytrain = pd.read_csv('data/ytrain.csv')
    ytest = pd.read_csv('data/ytest.csv')

    x_train = xtrain['Comments']
    y_train = ytrain['Classification']
    x_test = xtest['Comments']
    y_test = ytest['Classification']

    bow = CountVectorizer(
        min_df = 0.01,
        max_df = 0.85,
        binary = False,
        ngram_range=(1,3)
    )
    bow.fit(x_train)
    x_train = bow.transform(x_train)
    bow_x = bow.transform(x)

    path_ = open('models/model.pkl', 'rb')
    predictor = pickle.load(path_)

    prediction_ = predictor.predict(bow_x)

    # return prediction_
    if prediction_ == 0:
        return f"The comment above is not racist"
    elif prediction_ == 1:
        return f"The comment above is very racist"


with gr.Blocks() as demo:
    name = gr.Textbox()
    output = gr.Textbox()
    generate_ = gr.Button('Generate Response')
    generate_.click(fn = test_for_racism, inputs = name, outputs = output)

# demo = gr.Interface(fn = test_for_racism, inputs = "text", outputs = gr.Number())

demo.launch(share = False)





# I hate all Asians, they sould be enslaved