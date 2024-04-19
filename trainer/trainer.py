from models.transformer import Transformer
from torch.optim import Adam
from torch import nn 
import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self, path):
        self.model = Transformer(dim_model=32, num_heads=2, dim_feedforward=256, num_encoder_layers=2, num_decoder_layers=2, dropout_p=0.1)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.path = path
        self.summary_writer = SummaryWriter(path)
    
    def train(self, data, epoch, num_epochs):
        self.model.train()
        train_loss = 0.0
        for batch in data:
            inputs = batch[0]
            self.optimizer.zero_grad()

            outputs, _ = self.model(inputs, inputs)
            loss = self.loss_fn(outputs, inputs)
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
                outputs, _ = self.model(inputs)
                loss = self.loss_fn(outputs, inputs)
                val_loss += loss.item()
                self.summary_writer.add_scalar('Validation Loss', val_loss, epoch)

        val_loss /= len(data)

        print(f'Epoch [{epoch + 1}/{num_epochs}],Val Loss: {val_loss:.4f}')



    def save(self):
        torch.save(self.model.state_dict(), self.path)
        self.summary_writer.close()