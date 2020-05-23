import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import normalize
from tqdm import tqdm
from datetime import datetime


class SOM:
    """SOM with 2d grid"""

    _data = None
    _nUnits = 0
    _map_width = 0
    _unit_weights = None

    def __init__(self, data, map_width):
        """
        Initialise SOM
        :param data: 2D Data matrix. Data points arranged row-wise
        :param map_width: Width of the grid in the SOM algorithm
        """
        self._data = data
        self._nUnits = map_width ** 2
        self._map_width = map_width
        self._unit_weights = normalize(np.random.uniform(0, 1, (self._nUnits, data.shape[1])), norm="l2")

    def train(self, epochs=100, start=None, learning_rate=0.2):
        """
        Train the units in our map
        :param epochs:
        :param start:
        :param learning_rate:
        :return:
        """

        '''Aet up an exponentially decreasing neighbourhood size'''
        if start is None:
            start = self._map_width

        exponents = np.arange(epochs + 1)
        end_neighborhood = 0.5
        neighbour_distances = (np.floor(start * ((end_neighborhood/start)**(1/exponents[-1]))**exponents)).astype(int)

        '''Start training'''
        for epoch in tqdm(range(len(neighbour_distances))):
            '''Shuffle the data order'''
            data_order = np.arange(self._data.shape[0])
            np.random.shuffle(data_order)
            '''Loop through each data point and alter the units depending on the closest unit'''
            for data_index in data_order:
                winning_unit = self._get_winning_unit(data_index)  # Euclidean distance

                # update winning unit
                self._unit_weights[winning_unit] += learning_rate * (self._data[data_index] - self._unit_weights[winning_unit])

                # Find neighbours
                neighbours = []
                for neighbour_x in range(max(0, winning_unit % int(self._map_width) - neighbour_distances[epoch]),
                                         min(self._nUnits,
                                             winning_unit % int(self._map_width) + neighbour_distances[
                                                 epoch]) + 1):

                    if 0 <= neighbour_x < self._map_width:
                        for neighbour_y in range(-(neighbour_distances[epoch] - abs(
                                winning_unit % int(self._map_width) - neighbour_x)),
                                                 neighbour_distances[epoch] - abs(
                                                     winning_unit % int(self._map_width) - neighbour_x) + 1):
                            y_coord = neighbour_y + math.floor(winning_unit / self._map_width)
                            if 0 <= y_coord < self._map_width:
                                x_coord = neighbour_x
                                neighbours.append(x_coord + y_coord * self._map_width)

                # update neighbouring units
                for neighbour_index in neighbours:
                    if neighbour_index == winning_unit:
                        continue
                    neighbour_index_mod = neighbour_index % self._nUnits
                    self._unit_weights[neighbour_index_mod] += learning_rate * (
                            self._data[data_index] - self._unit_weights[neighbour_index_mod])

    def label_nodes(self, labels, method="d2u"):
        """
        Label the units with the provided data and their labels
        :param data: The input data
        :param labels: Label of the data
        :param method: Method for labeling data. 'd2u' will label each unit with a specific label if the unit is
        the closest one to the data point with that specific label. 'u2d' will label each unit to the same label as
        the closest data point. The distances are measured in euclidean space.
        :return: A matrix with the labels for each unit in the map.
        """

        '''Create return matrix'''
        unit_labels = [[] for _ in range(self._nUnits)]

        '''Label units'''
        if method == "d2u":
            for data_index in range(self.data.shape[0]):
                winning_unit = self._get_winning_unit(data_index)
                unit_labels[winning_unit].append(labels[data_index])
        else:
            for unit_index in range(self._nUnits):
                closest_data_point = self._get_closest_point(unit_index)
                unit_labels[unit_index].append(labels[closest_data_point])

        return unit_labels

    def _get_winning_unit(self, data_index):
        """
        Find closest unit to a specific data point
        :param data_index: the index of the data point
        :return: The index of the closest unit
        """
        distances = np.linalg.norm((self._unit_weights - self._data[data_index]), axis=1)
        return distances.argmin()

    def _get_closest_point(self, unit_index):
        """
        Find closest data point to a unit
        :param unit_index: The index of the unit
        :return: The index of the closest data point
        """
        distances = np.linalg.norm((self._data - self._unit_weights[unit_index]), axis=1)
        return distances.argmin()

    def plot_map(self, labels, label_name, data_point_name="input space", method="d2u", save_file=True):
        """
        Plot the units and label them with the provided labels
        :param labels: Label of the data points
        :param label_name: Label name for each label. These will be plotted.
        :param data_point_name: Name of the input space
        :param method: Method used when labeling data points. See self.label_nodes for more info.
        :param save_file: Boolean value indicating if we want to save the plotted map or show it in a window.
        :return:
        """

        ''' Label the units '''
        node_labels = self.label_nodes(labels, method)

        '''Fill a matrix representing our final labels for the units'''
        node_matrix_labels = np.full(len(node_labels), -1)
        for list_index in range(len(node_labels)):
            list_ = node_labels[list_index]
            if len(list_) == 0:
                continue
            most_frequent_element = max(set(list_), key=list_.count)
            node_matrix_labels[list_index] = most_frequent_element

        '''Plot labeled matrix and show or save to file'''
        dpi = 200
        plt.figure(figsize=(300 * self._map_width / dpi, 100 * self._map_width / dpi), dpi=dpi)
        for unit_index in range(self._nUnits):
            if node_matrix_labels[unit_index] != -1:
                unit_label = node_matrix_labels[unit_index]
                label_print = label_name[unit_label]
                x_coord = float(unit_index % self._map_width)
                y_coord = float(math.floor(unit_index / self._map_width))
                plt.annotate(label_print, (x_coord, y_coord))


        plt.xlim((-1, self._map_width))
        plt.ylim((-1, self._map_width))
        plt.title("Topological mapping to {}x{} grid with respect to {}"
                  .format(self._map_width, self._map_width, data_point_name))
        plt.xlabel("Grid x-axis")
        plt.ylabel("Grid y-axis")

        if save_file:
            now = datetime.now()
            now_str = now.strftime("%Y_%m_%d_%H_%M_%S")
            plt.savefig("results/{}".format(now_str + '.png'), dpi=dpi, bbox_inches='tight')
        else:
            plt.show()

        return
