from dataLoader.dataController import DataController
from trainer.trainer import Trainer
from trainer.test import Test

data_path = ['/home/wizav/src/timeseries_embedding/data/clippedregunathapuram-VH (1).tif',
             '/home/wizav/src/timeseries_embedding/data/clippedregunathapuram-VV (1).tif']
output_path = '/home/wizav/src/timeseries_embedding/data/'

is_test_train_split = True

# TODO Use cfgNode to supply args
# TODO better logs
# TODO remove pytorch and actual code
# Todo Make it importable in other functions


def train():

    train_data, val_data = DataController(data_path, is_test_train_split=is_test_train_split, batch_size=64)()

    trainer = Trainer(output_path)

    num_epochs = 50
    for epoch in range(num_epochs):
        trainer.train(train_data, epoch, num_epochs)
        if is_test_train_split:
            trainer.val(val_data, epoch, num_epochs)
    
    trainer.save()


if __name__=='__main__':
    
    train()
    