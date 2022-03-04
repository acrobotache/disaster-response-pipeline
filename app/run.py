import json
import operator
import re
import unicodedata
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import plotly
from bs4 import BeautifulSoup
from flask import Flask
from flask import render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

nltk.download(['wordnet', 'punkt', 'stopwords'])

app = Flask(__name__)


def tokenize(text):
    """
    Summary: utility function to pre-processes the text
    
    Parameters:
      text(str): the message
    
    Returns:
      lemmatized_text(list of str): a list of the root form of the words
    """
    # remove URLs
    patterns = [r"http\S+", r"www\S+"]
    for pattern in patterns:
        text = re.sub(pattern, "", text)

    # strip html tags    
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # remove accented charatcters
    text = (unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore"))

    # remove special characters and digits
    pattern = r"[^a-zA-Z0-9]"
    text = re.sub(pattern, " ", text.lower())

    # tokenize text
    word_tokens = word_tokenize(text)

    # remove english stop words
    stop_words_list = stopwords.words("english")
    words = [token for token in word_tokens if token not in stop_words_list]

    # lemmatization to get root words
    lemmatized_text = [WordNetLemmatizer().lemmatize(w) for w in words]

    return lemmatized_text


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponseTable', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    tokenized_words = list()
    for text in df['message'].values:
        tokenized_words.extend(tokenize(text))
    word_count_dict = Counter(tokenized_words)
    sorted_word_count_dict = dict(sorted(word_count_dict.items(), key=operator.itemgetter(1), reverse=True))
    top_10_words_dict = dict(list(sorted_word_count_dict.items())[:10])
    words = list(top_10_words_dict.keys())
    count_proportion = 100 * np.array(list(top_10_words_dict.values())) / df.shape[0]

    category_freq = df.iloc[:, 4:].sum()
    top_categories = category_freq.sort_values(ascending=False)[:10]
    top_category_names = list(top_categories.index)

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=words,
                    y=count_proportion
                )
            ],
            'layout': {
                'title': 'Top 10 Frequent Words',
                'yaxis': {
                    'title': 'Frequency(in Percentage)'
                },
                'xaxis': {
                    'title': 'Top 10 words'
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_category_names,
                    y=top_categories
                )
            ],

            'layout': {
                'title': 'Top 10 Frequent Categories',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
