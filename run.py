from dataLoader.dataController import DataController
from trainer.trainer import Trainer

data_path = '/home/wizav/src/timeseries_embedding/data/sample.npy'
output_path = '/home/wizav/src/timeseries_embedding/data/'

is_test_train_split = True

def train():

    train_data, val_data = DataController(data_path, is_test_train_split=is_test_train_split, batch_size=32)()

    trainer = Trainer(output_path)

    num_epochs = 5000
    for epoch in range(num_epochs):
        trainer.train(train_data, epoch, num_epochs)
        if is_test_train_split:
            trainer.val(val_data, epoch, num_epochs)
    
    trainer.save()







if __name__=='__main__':
    
    train()
    