from utils import convert_tensor_to_dataset, convert_dataset_to_dataloader
import torch

class Dataloader:
    def __init__(self, batch_size=32, shuffle = True):        
        self.batch_size = batch_size
        self.shuffle = shuffle

    def non_zero_filter(self, data):
        # Assuming your data consists of a data tensor and a label tensor
        return data[torch.count_nonzero(data, dim=1) > 0]

    def __call__(self, data):
        dataset = convert_tensor_to_dataset(data)
        return convert_dataset_to_dataloader(dataset=dataset, batch_size=self.batch_size, shuffle = self.shuffle)
    
