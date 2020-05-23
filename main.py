import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from SOM import SOM
from sklearn.preprocessing import normalize


class WordVectors:
    words = None

    def __init__(self, glove_data_file):
        self.words = pd.read_csv(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE,
                                 na_values=None, keep_default_na=False)
        return

    def get_vectors(self, words):
        words = [x.lower() for x in words]
        return self.words.loc[words]


def main():
    """Main function"""

    '''Load word vectors'''
    print("Reading data")
    filename = 'data/glove.6B.50d.txt'
    word_vectors = WordVectors(filename)

    '''Identify the n most frequent words in the English language'''
    n = 500  # Number of words to load
    filename_word_frequency = 'data/most_frequent_words/google-10000-english.txt'
    word_frequency = pd.read_csv(filename_word_frequency, sep=" ", header=None)
    most_frequent = word_frequency[:n]

    t = most_frequent.values.T[0]
    most_frequent_vectors = word_vectors.get_vectors(t)

    data = most_frequent_vectors.to_numpy()
    # data = normalize(data, norm="l2")
    i2w = most_frequent_vectors.index.to_list()

    '''Initialize SOM'''


    som = SOM(data, 20)

    '''Train SOM'''
    som.train(epochs=500, start=20, learningRate=0.2)

    '''Visualize'''
    som.plotMap(data, np.arange(len(i2w)), i2w, "word vector", method="u2d")

    print("Done")
    return


if __name__ == '__main__':
    main()
