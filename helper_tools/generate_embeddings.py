import torch
import numpy as np
from models.LSTM_autoencoder import LSTMAutoencoder


# Define the function to load the model and make predictions
def generate_embeddings(model_path, data_path, output_path):
  # Load the model
  model = LSTMAutoencoder()
  model.load_state_dict(torch.load(model_path))
  model.eval()  # Set the model to evaluation mode
  model.to('cuda')
  # Load the data (assuming it's a single NumPy array)
  data = np.load(data_path)

  # Convert data to a PyTorch tensor if necessary
  if not isinstance(data, torch.Tensor):
    data = torch.from_numpy(data).float().to('cuda')

  # Make predictions
  with torch.no_grad():  # Disable gradient calculation for efficiency
    encoded, _ = model(data)

  # Convert outputs back to NumPy array if necessary
  if isinstance(encoded, torch.Tensor):
    encoded = encoded.detach().cpu().numpy()

  # Save the outputs to a new NumPy file
  np.save(output_path, encoded)
  print(f"Output saved to: {output_path}")
  return encoded

if __name__=='__main__':

    # Example usage (replace paths with your actual file paths)
    model_path = "../data/model.pth"
    data_path = "../data/sample.npy"
    output_path = "../data/encoded_data.npy"

    embedded_data = generate_embeddings(model_path, data_path, output_path)


