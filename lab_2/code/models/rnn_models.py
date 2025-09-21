import torch
import torch.nn as nn
import copy

vocab_length = 500002
embed_len = 300
hidden_dim = 50
n_layers=3
num_classes = 15


class CreateRNN(nn.Module):
    def __init__(self, rnn_model):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_length, embedding_dim=embed_len)
        self.rnn_model = rnn_model(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch) # (batch_size, sequency_len, vector_len)
        output, _ = self.rnn_model(embeddings)
        return self.linear(output[:,-1])

class BlockGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_r = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.b_r = nn.Parameter(torch.Tensor(hidden_size))
        self.W_z = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.b_z = nn.Parameter(torch.Tensor(hidden_size))
        self.W_h = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))
        self.sigma = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, embeddings, hidden):
        seq_len = embeddings.size(1)
        outputs = []

        for t in range(seq_len):
            x_t = embeddings[:, t, :]  # (batch, vector)

            hidden_x_t = torch.cat([hidden, x_t], dim=1)  # (batch, vector + hidden)

            r_t = self.sigma(torch.matmul(hidden_x_t, self.W_r) + self.b_r)  # (batch, hidden)
            z_t = self.sigma(torch.matmul(hidden_x_t, self.W_z) + self.b_z)  # (batch, hidden)

            h_tilde = self.tanh(torch.matmul(torch.cat([r_t * hidden, x_t], dim=1), self.W_h) + self.b_h)  # (batch, hidden)

            hidden = (1 - z_t) * hidden + z_t * h_tilde  # (batch, hidden)
            outputs.append(hidden.unsqueeze(1))  # (batch, 1, hidden)

        outputs = torch.cat(outputs, dim=1)  # (batch, seq, hidden)
        return outputs, hidden
    
class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super().__init__()
        self.gru = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gru.append(BlockGRU(input_size, hidden_size))
            else:
                self.gru.append(BlockGRU(hidden_size, hidden_size))

    def forward(self, embeddings):
        batch_size = embeddings.size(0)
        hidden = torch.zeros((batch_size, hidden_dim))

        output = embeddings
        for layer in self.gru:
            output, hidden = layer(output, hidden)

        return output, hidden