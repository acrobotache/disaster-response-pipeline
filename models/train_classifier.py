# import libraries
import pickle
import re
import sys
import unicodedata

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download(['wordnet', 'punkt', 'stopwords'])


def load_data(database_filepath):
    """
        Summary: utility function to load saved data from SQLite database

        Parameters:
            database_filepath(str): database file path

        Returns:
            X (DataFrame) : dataframe containing messages(features)
            Y (DataFrame) : target categories
            category (list of str) : target labels list
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponseTable', engine)
    X = df['message']  # message column
    Y = df.iloc[:, 4:]  # classification categories(labels)
    return X, Y, list(Y.columns)


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


def build_model():
    """
        Summary: utility function to build machine learning pipeline to classifiy disaster messages

        Returns:
            model_cv: classification GridSearchCV object (model)
    """
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    model_parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 60, 70]
    }

    model_cv = GridSearchCV(model, param_grid=model_parameters)
    return model_cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        Summary: utility function to evaluate the model and print the f1 score, precision and recall for each output category

        Parameters:
            model: trained classification model
            X_test: test messages/features to be evaluated
            Y_test: test categories for respective test massages
            category_names(list): list of names of categories
    """
    Y_pred = model.predict(X_test)
    counter = 0
    for category in category_names:
        print('Classification Report for Output Category {}: {}'.format(counter + 1, category))
        print(classification_report(Y_test[category], Y_pred[:, counter]))
        counter = counter + 1
    print('The model accuracy is {:.4f}'.format((Y_pred == Y_test.values).mean()))


def save_model(model, model_filepath):
    """
        Summary: utility function to saved trained classification model

        Parameters:
            model: trained classification model
            model_filepath(str): pickle file path where model is to be saved
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
