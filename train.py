from dataLoader.dataController import DataController
from trainer.trainer import Trainer

data_path = '/media/wizav/DATA/companies/farmwiseai/crop_monitoring/Clustering_ts_dataset_1/Regunathapuram/runs/merged_data.npy'
model_path = '/media/wizav/DATA/companies/farmwiseai/crop_monitoring/Clustering_ts_dataset_1/Regunathapuram/runs/model.pth'
is_test_train_split = True

def train():

    train_data, val_data = DataController(data_path, is_test_train_split=is_test_train_split, batch_size=32)()

    trainer = Trainer(model_path)

    num_epochs = 50 
    for epoch in range(num_epochs):
        trainer.train(train_data, epoch, num_epochs)
        if is_test_train_split:
            trainer.val(val_data, epoch, num_epochs)
    
    trainer.save()







if __name__=='__main__':
    
    train()
    