# Created by lan at 2020/7/6
import csv

import numpy as np


def load_configuration_files():
    """
    Load the q-learning configuration files
    :return: the loaded reward matrix, and the list of action names
    """
    with open('../resources/reward-matrix.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        r_matrix = np.array(list(csv_reader), dtype='float32')

    with open('../resources/actions.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        actions_name = np.transpose(np.array(list(csv_reader)))[0]

    return r_matrix, actions_name