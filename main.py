import numpy as np
import argparse
import pandas as pd
import csv

from SOM import SOM
from sklearn.preprocessing import normalize


class WordVectors:
    """
    Class for loading our word vectors accessing them
    """

    words = None

    def __init__(self, glove_data_file):
        """
        Initialize by reading data file and storing the pre-trained word vectors in a dataframe
        :param glove_data_file:
        """
        self.words = pd.read_csv(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE,
                                 na_values=None, keep_default_na=False)

    def get_vectors(self, words):
        """
        Get requested word vectors
        :param words: List of the words we want word vectors of
        :return: Word vectors of the requested words (if they exist)
        """
        words = [x.lower() for x in words]
        return self.words.loc[self.words.index.intersection(words)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embedding representation with SOM')
    parser.add_argument('-f', '--file', type=str, help='A textual file containing word vectors',
                        default='data/glove.6B.50d.txt')
    parser.add_argument('-mcwf', '--most-common-words-file', type=str,
                        help='A textual file containing a list of the most common words '
                             'or the words we wish to include in the SOM',
                        default='data/most_frequent_words/google-10000-english-no-swears.txt')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs in SOM algorithm')
    parser.add_argument('-gw', '--grid-width', default=10, type=int, help='Grid width of the map in the SOM algorithm')
    parser.add_argument('-nr', '--neighborhood-range', default=10, type=int,
                        help='Determined the initial neighborhood size in SOM algorithm. This includes all '
                             'neighbours with a manhattan distance of ns from the focus unit')
    parser.add_argument('-ds', '--data-size', default=200, type=int, help='Number of data points to include')
    parser.add_argument('-lr', '--learning-rate', default=0.2, type=float, help='Learning rate in SOM algorithm')
    parser.add_argument('-lu', '--label-unit', default='u2d', type=str, choices=['d2u', 'u2d'],
                        help='Method for labeling the units')
    args = parser.parse_args()

    '''Load word vectors'''
    filename = args.file
    print("Reading data: {}".format(filename))
    word_vectors = WordVectors(filename)

    '''Identify the n most frequent words in the English language'''
    n = args.data_size  # Number of words to load
    filename_word_frequency = args.most_common_words_file
    word_frequency = pd.read_csv(filename_word_frequency, sep=" ", header=None)
    most_frequent = word_frequency[:n]

    most_frequent_vectors = word_vectors.get_vectors(most_frequent.values.T[0])

    data = most_frequent_vectors.to_numpy()
    data = normalize(data, norm="l2")
    i2w = most_frequent_vectors.index.to_list()

    '''Initialize SOM'''
    som = SOM(data, args.grid_width)

    '''Train SOM'''
    som.train(epochs=args.epochs, start_range=args.neighborhood_range, learning_rate=args.learning_rate)

    '''Visualize'''
    som.plot_map(np.arange(len(i2w)), i2w, "word vector", method=args.label_unit, save_file=True)

    print("Done")
