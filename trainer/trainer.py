from models.transformer import Transformer
from torch.optim import Adam
from torch import nn 
import torch
from models.LSTM_autoencoder import LSTMAutoencoder
from torch.utils.tensorboard import SummaryWriter
import os


class Trainer():
    def __init__(self, path, model_name='LSTM_autoencoder'):
        self.model = model_map[model_name]()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.model_path = os.path.join(path, 'model.pth')
        self.summary_writer = SummaryWriter(os.path.join(path, 'runs'))
    
    def train(self, data, epoch, num_epochs):
        self.model.train()
        train_loss = 0.0
        for batch in data:
            inputs = batch[0]
            self.optimizer.zero_grad()
            target = inputs
            encoded_output, outputs = self.model(inputs)
            loss = self.loss_fn(outputs, target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.summary_writer.add_scalar('Training Loss', loss.item(), epoch)
        train_loss /= len(data)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}],Train Loss: {train_loss:.4f}')
              

    def val(self, data, epoch, num_epochs):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in data:
                inputs = batch[0]
                _, outputs = self.model(inputs)
                loss = self.loss_fn(outputs, inputs)
                val_loss += loss.item()
                self.summary_writer.add_scalar('Validation Loss', val_loss, epoch)

        val_loss /= len(data)

        print(f'Epoch [{epoch + 1}/{num_epochs}],Val Loss: {val_loss:.4f}')



    def save(self):
        torch.save(self.model.state_dict(), self.model_path)
        self.summary_writer.close()


model_map = {
    'LSTM_autoencoder': LSTMAutoencoder
}