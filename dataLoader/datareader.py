from typing import Any
from utils import read_data, convert_data_to_tensor

class DataReader:
    def __init__(self, path) -> None:
        self.path = path

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.data = read_data(self.path)
        return self
    
    @property
    def to_tensor(self):
        # Exception to be handled if data is not available
        data =  convert_data_to_tensor(self.data).to('cuda')
        return data
    