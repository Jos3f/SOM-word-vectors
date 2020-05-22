import numpy as np
import pandas as pd
import csv


class WordVectors:
    words = None

    def __init__(self, glove_data_file):
        # https://stackoverflow.com/a/45894001
        self.words = pd.read_csv(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE, na_values=None, keep_default_na=False)
        return

    def get_vectors(self, words):
        # words = [x.lower() for x in words]
        return self.words.loc[words]


def main():
    """Main function"""

    '''Load word vectors'''
    filename = 'data/glove.6B.50d.txt'
    word_vectors = WordVectors(filename)

    '''Identify most frequent words'''
    filename_word_frequency = 'data/most_frequent_words/google-10000-english.txt'
    word_frequency = pd.read_csv(filename_word_frequency, sep=" ", header=None)
    most_frequent = word_frequency[:200]

    t = most_frequent.values.T[0]
    most_frequent_vectors = word_vectors.get_vectors(t)

    '''Initialize SOM'''

    '''Train SOM'''

    '''Visualize'''

    print("Done")
    return


if __name__ == '__main__':
    main()
