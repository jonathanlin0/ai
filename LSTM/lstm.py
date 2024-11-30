# adds parent directory to python path to import other utility modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from data.spam_text import SpamTextData
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        """
        params
            input_dim (int): the number of expected features in the input x
            hidden_dim (int): the number of features in the hidden state h
            layer_dim (int): the number of RNN layers
            output_dim (int): the number of output features
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to("cuda")

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to("cuda")

        # Detach the gradients to prevent backpropagation through time
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs for the fully connected layer
        out = self.fc(out[:, -1, :])
        out = self.fc_2(out)
        return out

if __name__ == "__main__":
    train_loader, test_loader = SpamTextData.get_data(batch_size=64)

    # define the RNN model
    # vocab size is the total number of unique words in the dataset
    # vocab_size = 13580
    # embed_size = 128
    hidden_size = 512
    output_size = 2
    model = LSTMModel(input_dim=1, hidden_dim=hidden_size, layer_dim=2, output_dim=output_size).to("cuda")

    criterion = nn.CrossEntropyLoss()
    # both produce eh results. same as using RNN
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 100
    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for texts, labels in train_loader:
            texts = texts.unsqueeze(2).to("cuda")
            labels = labels.to("cuda")
            # texts, labels = texts.to("cuda"), labels.to("cuda")
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(train_loader))
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
    
    # evaluate performance of model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to("cuda"), labels.to("cuda")
            texts = texts.unsqueeze(2)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig("LSTM/lstm_loss.png")
    plt.show()