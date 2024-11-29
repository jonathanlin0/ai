
# adds parent directory to python path to import other utility modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data.spam_text import SpamTextData

class MultiLayerRNN(nn.Module):
    """
    Implementation of a simple RNN model.
    Not exactly a RNN cause it takes the last hidden state and passes it to the linear layer.
    """
    def __init__(self, input_size, embed_size, hidden_size, output_size, num_rnn_layers=1):
        """
        params
            input_size (int): the number of expected features in the input x
            embed_size (int): the number of features in the embedding layer
            hidden_size (int): the number of features in the hidden state h
            output_size (int): the number of output features
        """
        super(MultiLayerRNN, self).__init__()
        # embedding layer simply converts the input into a dense vector of fixed size
        # nn.Embedding is used when dealing with categorical data (labels are 0,1,2,...) over nn.Linear, but they're rly similar. have to do more research into the differences later
        self.num_layers = num_rnn_layers
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
    
    def forward(self, x):
        x = self.embedding(x)
        # h0 is the first hidden state in the RNN, initialized to all zeros
        # h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    

if __name__ == "__main__":

    train_loader, test_loader = SpamTextData.get_data(batch_size=64)

    # define the RNN model
    # vocab size is the total number of unique words in the dataset
    vocab_size = 13580
    embed_size = 128
    hidden_size = 256
    output_size = 2
    model = MultiLayerRNN(vocab_size, embed_size, hidden_size, output_size).to("cuda")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to("cuda"), labels.to("cuda")
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
    plt.savefig("RNN/multi_layer_rnn_loss.png")
    plt.show()