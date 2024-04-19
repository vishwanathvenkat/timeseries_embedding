from utils import convert_tensor_to_dataset, convert_dataset_to_dataloader

class Dataloader():
    def __init__(self, batch_size=32, shuffle = True):        
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, data):
        dataset = convert_tensor_to_dataset(data)
        return convert_dataset_to_dataloader(dataset, self.batch_size, self.shuffle)
    
