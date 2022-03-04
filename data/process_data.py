# import libraries

import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
        Summary: utility function to load and merge message and category data

        Parameters:
            messages_filepath(str): the messase data file path
            categories_filepath(str): the category data file path

        Returns:
            df(DataFrame): a dataframe containing messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='inner')
    return df


def clean_data(df):
    """
        Summary: utility function to clean the dataframe

        Parameters:
            df(DataFrame): a dataframe containing messages and categories to be cleaned

        Returns:
            df(DataFrame): a cleaned dataframe containing messages and categories
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.head(1)
    category_column_names = row.applymap(lambda x: x[:-2]).iloc[0, :]
    category_column_names = category_column_names.tolist()
    categories.columns = category_column_names
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1, join='inner')
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
        Summary: utility function to save a dataframe in database

        Parameters:
            df(DataFrame): a cleaned dataframe containing messages and categories
            database_filename(str): database file name
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponseTable', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
