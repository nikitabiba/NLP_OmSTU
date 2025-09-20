import torch
import torch.nn as nn
import copy

vocab_length = 500002
embed_len = 300
hidden_dim = 50
n_layers=1
num_classes = 15


class CreateRNN(nn.Module):
    def __init__(self, rnn_model):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_length, embedding_dim=embed_len)
        self.rnn_model = rnn_model(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch) # (batch_size, sequency_len, vector_len)
        output, _ = self.rnn_model(embeddings) # (n_layers, batch_size, hidden_dim)
        return self.linear(output[:,-1])

class BlockGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_r = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.b_r = nn.Parameter(torch.Tensor(hidden_size))
        self.W_z = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.b_z = nn.Parameter(torch.Tensor(hidden_size))
        self.W_h = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))
        self.sigma = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, embeddings, hidden):
        for i in range(len(embeddings)):
            for j in range(len(embeddings[i])):
                r_t = self.sigma(torch.matmul(self.W_r, torch.cat(hidden, embeddings[i][j])) + self.b_r)
                z_t = self.sigma(torch.matmul(self.W_z, torch.cat(hidden, embeddings[i][j])) + self.b_z)
                h__t = self.tanh(torch.matmul(self.W_h, torch.cat(torch.mul(r_t, hidden), embeddings[i][j])) + self.b_h)
                h_t = torch.add(torch.mul(1 - z_t, hidden), torch.mul(z_t, h__t))
                output, hidden = copy.deepcopy(h_t), copy.deepcopy(h_t)
        return output, hidden
    
class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super().__init__()
        self.gru = []
        for i in range(num_layers):
            if i == 0:
                self.gru.append(BlockGRU(input_size, hidden_size))
            else:
                self.gru.append(BlockGRU(hidden_size, hidden_size))

    def forward(self, embeddings):
        for i in range(len(self.gru)):
            if i == 0:
                hidden = torch.zeros((hidden_dim))
            output, hidden = self.gru[i](embeddings, hidden)
        return output, hidden