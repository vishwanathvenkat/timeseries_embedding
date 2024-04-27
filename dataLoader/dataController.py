from typing import Any
from dataLoader.datareader import DataReader
from utils import split_data
from dataLoader.dataLoader import Dataloader
import numpy as np
from utils import convert_data_to_tensor

class DataController:
    def __init__(self, paths, is_test_train_split, batch_size, type='raster') -> None:
        self.data = self.stack_data([DataReader(path, type).get_data for path in paths])

        self.is_test_train_split = is_test_train_split
        self.dataLoader = Dataloader(batch_size, shuffle=True)
    # def get_corresponding_indices(self):
    #     data = self.dataReader.data
    #     x, y = np.meshgrid(range(1, data.shape[0]+1), range(1, data.shape[1]+1), indexing='ij')
    #     list_of_indices = list(zip(x.ravel(), y.ravel()))
    #     return list_of_indices
    def __call__(self) -> Any:
        data = self.get_tensor_data
        if self.is_test_train_split:
            train, val = split_data(data)
            train_dataloader = self.dataLoader(train)
            val_dataloader = self.dataLoader(val)
        else:
            train_dataloader = self.dataLoader(data)
            val_dataloader = None
        return train_dataloader, val_dataloader
    def stack_data(self, data_list):
        return np.stack(data_list, axis=-1)

    @property
    def get_tensor_data(self):
        # Exception to be handled if data is not available
        data =  convert_data_to_tensor(self.data).to('cuda')
        return data
        