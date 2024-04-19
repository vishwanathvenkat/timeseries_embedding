from typing import Any
from dataLoader.datareader import DataReader
from utils import split_data
from dataLoader.dataLoader import Dataloader
class DataController:
    def __init__(self, path, is_test_train_split, batch_size) -> None:
        self.dataReader = DataReader(path)()
        self.is_test_train_split = is_test_train_split
        self.dataLoader = Dataloader(batch_size, shuffle=True)
    
    def __call__(self) -> Any:
        data = self.dataReader.to_tensor
        if self.is_test_train_split:
            train, val = split_data(data)
            train_dataloader = self.dataLoader(train)
            val_dataloader = self.dataLoader(val)
        else:
            train_dataloader = self.dataLoader(data)
            val_dataloader = None
        return train_dataloader, val_dataloader