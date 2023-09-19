from torch import nn
import torch
from padding import vocab

html_label = ["Safe","Reflect","Storage"]
device = torch.device("cpu")

embed_len = 50
hidden_dim = 75
n_layers=1

class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_len)
        self.lstm = nn.LSTM(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, len(html_label))
        self.relu = nn.ReLU()

    def forward(self, X_batch):
        X_batch = X_batch.to(device).to(torch.int64)
        embeddings = self.embedding_layer(X_batch)
        hidden, carry = torch.randn(n_layers, len(X_batch), hidden_dim), torch.randn(n_layers, len(X_batch), hidden_dim)
        hidden = hidden.to(device)
        carry = carry.to(device)
        output, (hidden, carry) = self.lstm(embeddings, (hidden, carry))
        #output = F.softmax(output)
        output = self.relu(output)
        return self.linear(output[:,-1])