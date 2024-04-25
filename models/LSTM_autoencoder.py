from torch import nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
      super(LSTMEncoder, self).__init__()
      self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
      self.linear = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
      _, (hidden, _) = self.lstm(x)
      return self.linear(hidden.squeeze(0))


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, sequence_length):
        super(LSTMDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Repeat latent vector to match sequence length
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        x, _ = self.lstm(x)
        return self.linear(x)


class LSTMAutoencoder(nn.Module):
  def __init__(self, input_dim=1, latent_dim=512, hidden_dim=1, sequence_length=20):
    super(LSTMAutoencoder, self).__init__()
    self.encoder = LSTMEncoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
    self.decoder = LSTMDecoder(latent_dim=latent_dim,
                               output_dim=input_dim,
                               hidden_dim=hidden_dim,
                               sequence_length=sequence_length)

  def forward(self, x):
    # Pass through encoder
    encoded_data = self.encoder(x)

    # Pass through decoder
    decoded_data = self.decoder(encoded_data)

    return encoded_data, decoded_data
