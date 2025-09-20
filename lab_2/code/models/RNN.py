import torch
import torch.nn as nn

vocab_length = 500002
embed_len = 300
hidden_dim = 20
n_layers=1
num_classes = 15


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_length, embedding_dim=embed_len)
        self.rnn = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn(embeddings, torch.randn(n_layers, len(X_batch), hidden_dim))
        return self.linear(output[:,-1])
