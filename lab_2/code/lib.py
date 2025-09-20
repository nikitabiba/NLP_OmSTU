from navec import Navec
import torch

navec = Navec.load("../models/navec_hudlit_v1_12B_500K_300d_100q.tar")

max_words = 50


def tokenizer(text):
    result = ''
    for char in text.lower():
        if 'а' <= char <= 'я' or char == 'ё' or char == ' ':
            result += char
    return result.strip().split()

def vocab(tokens):
    indexes = []
    for token in tokens:
        if token in navec.vocab:
            indexes.append(navec.vocab[token])
        else:
            indexes.append(navec.vocab['<unk>'])
    return indexes

def vectorize_batch(batch):
    X, Y = list(zip(*batch))
    X = [vocab(tokenizer(text)) for text in X]
    X = [tokens+([0]* (max_words-len(tokens))) if len(tokens)<max_words else tokens[:max_words] for tokens in X]
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)