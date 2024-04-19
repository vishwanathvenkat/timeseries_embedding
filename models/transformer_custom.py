from torch import nn

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim):
        super(TransformerAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Replace the encoder, decoder with a transformer module
        # Encoder part
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim), num_layers=2)

        # Decoder part
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, input_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded