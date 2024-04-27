from torch import nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
      super(LSTMEncoder, self).__init__()
      self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
      self.linear = nn.Linear(hidden_dim, latent_dim)
      # TODO use sequential to merge the dimensions
    def forward(self, x):
      lstm_out, output = self.lstm(x)
      last_hidden = lstm_out[:, -1, :]  # Select the last hidden state from the last layer

      # Pass the hidden state through the linear layer
      return self.linear(last_hidden)



class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, sequence_length):
        super(LSTMDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True, num_layers=2)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Repeat latent vector to match sequence length
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        x, _ = self.lstm(x)
        return self.linear(x)


class LSTMAutoencoder(nn.Module):
  def __init__(self, input_dim=2, latent_dim=512, hidden_dim=64, sequence_length=24):
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


"""
Tips for better hyper parameter tuning:
To improve the convergence rate of your LSTM-based autoencoder in PyTorch, you can consider adjusting several hyperparameters and aspects of your model architecture. Here are some suggestions:

1. **Increase Hidden Dimension**: The `hidden_dim=1` seems quite low, especially given that your `latent_dim` is 512. This large discrepancy might be causing a bottleneck in your network, where the model is unable to capture enough complexity to effectively learn the underlying patterns in the data. Increasing the `hidden_dim` might help the model to capture more complex features and hence converge faster. Try increasing it to a higher value (e.g., 64, 128, or even 256) and observe the impact on training convergence.

2. **Adjust Learning Rate**: Sometimes, the choice of learning rate can significantly affect the speed of convergence. If the learning rate is too low, the model might take longer to converge; if it's too high, the model might overshoot the minima. Experiment with different learning rates. Additionally, consider using learning rate schedulers like `ReduceLROnPlateau` or `StepLR` to adjust the learning rate dynamically based on the training/validation loss.

3. **Batch Size**: Adjusting the batch size can also impact the training dynamics. A larger batch size provides a more accurate estimate of the gradient, but it can also lead to a smoother optimization landscape. On the other hand, a smaller batch size can help escape local minima but might make the training process noisier. Experiment with different batch sizes to find a balance that works for your specific case.

4. **Optimizer**: If you are using a basic optimizer like SGD, consider switching to more advanced optimizers like Adam or RMSprop, which are generally better for training deep neural networks like LSTMs. These optimizers adjust the learning rates dynamically and can lead to faster convergence.

5. **Number of Layers**: Adding more LSTM layers can help the model learn more complex patterns. However, this will also increase the computational complexity. If your current setup includes only one LSTM layer, consider adding one or two more layers.

6. **Regularization Techniques**: If overfitting is an issue (which might not be evident initially but can affect convergence), consider applying dropout or L2 regularization. Dropout can be particularly effective in recurrent networks by adding it between LSTM layers.

7. **Activation Functions**: Check the activation functions used in your network. Sometimes, changing the activation function (e.g., from `tanh` to `ReLU` or vice versa) can affect the training dynamics.

8. **Sequence Length and Model Architecture**: Although you mentioned not wanting to change the dataset, consider if the sequence length of 24 is optimal for the patterns in your data. Adjusting the sequence length can sometimes lead to better or worse performance depending on the context of the data.

9. **Early Stopping**: To avoid unnecessary training epochs and potential overfitting, implement early stopping in your training loop. This stops the training process once the model performance stops improving on a held-out validation set.

Implement these changes iteratively, testing each change's effect on the model's performance to understand which adjustments are most beneficial for your specific scenario.
"""
