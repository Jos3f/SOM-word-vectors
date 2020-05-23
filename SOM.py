import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
import time
import random
from sklearn.preprocessing import normalize
from tqdm import tqdm



# Taken from w2v.py
def load(fname):
    """
    Load the word2vec model from a file `fname`
    """
    try:
        with open(fname, 'r') as f:
            V, H = (int(a) for a in next(f).split())

            W, i2w, w2i = [None] * V, [], {}
            for i, line in enumerate(f):
                parts = line.split()
                word = parts[0].strip()
                w2i[word] = i
                W[i] = list(map(float, parts[1:]))
                i2w.append(word)

    except:
        print("Error: failing to load the model to the file")
    return W, w2i, i2w


class SOM:
    """SOM with 2d grid"""

    def __init__(self, data, map_width):
        self.data = data
        self.nUnits = map_width ** 2
        self.map_width = map_width
        # self.weights = np.random.uniform(0,1,(self.nUnits, dataDim))
        self.weights = normalize(np.random.uniform(0, 1, (self.nUnits, data.shape[1])), norm="l2")

    def train(self, epochs=100, start=None, learningRate=0.2):
        if start is None:
            start = self.map_width


        exponents = np.arange(epochs + 1)
        end_neighborhood = 0.5
        neighbourDistances = (np.floor(start * ((end_neighborhood/start)**(1/exponents[-1]))**exponents)).astype(int)

        for epoch in tqdm(range(len(neighbourDistances))):
            data_order = np.arange(self.data.shape[0])
            np.random.shuffle(data_order)
            for data_index in data_order:
                winning_unit = self._get_winning_unit(data_index)

                # update winner
                self.weights[winning_unit] += learningRate * (self.data[data_index] - self.weights[winning_unit])

                # Calculate neighbours
                neighbours = []
                # print("winning_unit: " + str(winning_unit))
                for neighbour_x in range(max(0, winning_unit % int(self.map_width) - neighbourDistances[epoch]),
                                         min(self.nUnits,
                                             winning_unit % int(self.map_width) + neighbourDistances[
                                                 epoch]) + 1):
                    # print("neighbour_x:" + str(neighbour_x))
                    if neighbour_x >= 0 and neighbour_x < self.map_width:
                        for neighbour_y in range(-(neighbourDistances[epoch] - abs(
                                winning_unit % int(self.map_width) - neighbour_x)),
                                                 neighbourDistances[epoch] - abs(
                                                     winning_unit % int(self.map_width) - neighbour_x) + 1):
                            y_coord = neighbour_y + math.floor(winning_unit / self.map_width)
                            if y_coord >= 0 and y_coord < self.map_width:
                                x_coord = neighbour_x
                                neighbours.append(x_coord + y_coord * self.map_width)
                # print("neighbours: ")
                # print(neighbours)

                # update neighbours
                for neighbour_index in neighbours:
                    if neighbour_index == winning_unit:
                        continue
                    neighbour_index_mod = neighbour_index % self.nUnits
                    # if random.uniform(0,1) > 0.5:
                    #     self.weights[neighbour_index_mod] += learningRate * (data[data_index] - self.weights[neighbour_index_mod])
                    self.weights[neighbour_index_mod] += learningRate * (
                            self.data[data_index] - self.weights[neighbour_index_mod])

    def label_nodes(self, data, labels, method="d2u"):
        unit_labels = [[] for _ in range(self.nUnits)]

        if method == "d2u":
            for data_index in range(data.shape[0]):
                # print(data_index)
                winning_unit = self._get_winning_unit(data_index)

                unit_labels[winning_unit].append(labels[data_index])
        else:
            for unit_index in range(self.nUnits):
                closest_data_point = self._get_closest_point(unit_index)
                unit_labels[unit_index].append(labels[closest_data_point])

        return unit_labels

    def _get_winning_unit(self, data_index):
        distances = np.linalg.norm((self.weights - self.data[data_index]), axis=1)
        return distances.argmin()

    def _get_closest_point(self, unit_index):
        distances = np.linalg.norm((self.data - self.weights[unit_index]), axis=1)
        return distances.argmin()

    def plotMap(self, mpvotes, labels, label_name, data_point_name, method="d2u"):
        node_labels = self.label_nodes(mpvotes, labels, method)

        print("Node labels: ")
        print(node_labels)

        node_matrix_labels = np.full(len(node_labels), -1)

        for list_index in range(len(node_labels)):
            list = node_labels[list_index]
            if len(list) == 0:
                continue
            most_frequent_element = max(set(list), key=list.count)
            node_matrix_labels[list_index] = most_frequent_element

        # print(node_matrix_labels)
        for row in range(self.map_width):
            print(node_matrix_labels[self.map_width * row:self.map_width * (row + 1)])

        for unit_index in range(self.nUnits):
            if node_matrix_labels[unit_index] != -1:
                unit_label = node_matrix_labels[unit_index]
                label_print = label_name[unit_label]
                # print(unit_index)
                # print(unit_label)
                # print(party_letter)
                x_coord = float(unit_index % self.map_width)
                y_coord = float(math.floor(unit_index / self.map_width))
                # print(x_coord)
                # print(y_coord)
                plt.annotate(label_print, (x_coord, y_coord))

        plt.xlim((-1, self.map_width))
        plt.ylim((-1, self.map_width))
        plt.title("Topological mapping to {}x{} grid with respect to {}"
                  .format(self.map_width, self.map_width, data_point_name))
        plt.xlabel("Grid x-axis")
        plt.ylabel("Grid y-axis")
        plt.show()


def main():
    # Load data files
    # mpvotes = np.loadtxt('data/votes.dat', dtype='float', delimiter=',').reshape((349,31))
    W, w2i, i2w = load("word_vectors.txt")
    W = np.array(W)

    # ind = np.argwhere(np.absolute(W).sum(axis=1) > 400**2).T[0]
    ind = np.argsort(- np.absolute(W).sum(axis=1)).T[:163]
    W = W[ind, :]
    i2w = np.array(i2w)[ind]

    W = normalize(W, norm="l2")

    # random_points = np.random.choice(W.shape[0], 350, replace=False)
    # W = W[random_points, :]
    # i2w = np.array(i2w)[random_points]

    som = SOM(W, 10)
    som.train(epochs=50, start=5, learningRate=0.2)

    som.plotMap(W, np.arange(len(i2w)), i2w, "word vector")




if __name__ == '__main__':
    main()
