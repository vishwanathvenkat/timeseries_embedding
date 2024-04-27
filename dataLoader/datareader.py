from typing import Any
from utils import read_data
import numpy as np


class DataReader:
    def __init__(self, path, type) -> None:
        self.path = path
        self.type = type
        self.data, self.list_indices_tuples = self.process()

    def process(self ) -> Any:
        data = read_data(self.path, self.type)
        true_false_matrix = self.filter_zero_values(data)
        list_indices_tuples = self.get_list_indices_tuple(true_false_matrix)
        shape = data.shape
        data = data.reshape(shape[0]*shape[1], shape[2])
        data = self.remove_zero_arrays(data)
        assert len(data)==len(list_indices_tuples)
        return data, list_indices_tuples

    def filter_zero_values(self, data):
        non_zero_rows = np.any(data, axis=2)
        return non_zero_rows

    def remove_zero_arrays(self, data):
        nonzero_rows_mask = np.any(data != 0, axis=1)
        # Use the mask to select rows with at least one nonzero element
        non_zero_data = data[nonzero_rows_mask]
        return non_zero_data

    def get_list_indices_tuple(self, keep_rows):
        true_indices = np.where(keep_rows)
        result_list = [(row + 1, col + 1) for row, col in zip(true_indices[0], true_indices[1])]
        return result_list

    @property
    def get_data(self):
        return self.data