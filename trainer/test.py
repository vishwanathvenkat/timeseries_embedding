import torch
from models.LSTM_autoencoder import LSTMAutoencoder

class Test:
    def __init__(self, path, model_name='LSTM_autoencoder'):
        self.model = model_map[model_name]().to('cuda')
        self.model.load_state_dict(torch.load(path+'/model.pth'))


    def test(self, data):
        self.model.eval()
        encoded_list = []
        with torch.no_grad():  # Disable gradient calculation for efficiency
            for batch in data:
                inputs = batch[0]
                encoded, _ = self.model(inputs)
                encoded = encoded.detach().cpu().numpy()
                encoded_list.extend(encoded)

        return encoded_list


model_map = {
    'LSTM_autoencoder': LSTMAutoencoder
}