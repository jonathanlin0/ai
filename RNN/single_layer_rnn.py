
# adds parent directory to python path to import other utility modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data.gen_sin_data import GenSinData


class SingleLayerRNN(nn.Module):
    """
    Implementation of a simple RNN model.
    Not exactly a RNN cause it takes the last hidden state and passes it to the linear layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SingleLayerRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.hidden_size = hidden_size
    
    def forward(self, x):
        batch_size = x.size(0)
        # h0 is the first hidden state in the RNN, initialized to all zeros
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    # define the RNN model
    input_size = 1
    hidden_size = 20
    output_size = 1
    model = SingleLayerRNN(input_size, hidden_size, output_size)

    # generate the data
    X, y = GenSinData.gen_data()
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X.unsqueeze(2))  # Add a dimension for input size
        loss = criterion(outputs, y.unsqueeze(2))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X.unsqueeze(2)).squeeze(2).numpy()

    # plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y[0].numpy(), label='True')
    plt.plot(predictions[0], label='Predicted')
    plt.legend()
    plt.show()