from navec import Navec
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

navec = Navec.load("../models/navec_hudlit_v1_12B_500K_300d_100q.tar")

max_words = 310


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

def train_val_test(epochs, train_loader, val_loader, test_loader, model, device):
    loss_model = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                            mode='min',
                                                            factor=0.1,
                                                            patience=7,
                                                            threshold=1e-4,
                                                            threshold_mode='rel',
                                                            cooldown=3,
                                                            min_lr=0,
                                                            eps=1e-8,
                                                            )

    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    train_precision, train_recall, train_f1 = [], [], []
    val_precision, val_recall, val_f1 = [], [], []
    list_ = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = []
        all_train_preds, all_train_targets = [], []
        train_loop = tqdm(train_loader, leave=False)

        for x, targets in train_loop:
            x = x.to(device)
            targets = targets.to(device)

            pred = model(x)
            loss = loss_model(pred, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_train_loss.append(loss.item())

            preds = pred.argmax(dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_targets.extend(targets.cpu().numpy())

            train_loop.set_description(
                f"Epoch {epoch + 1}/{epochs};   loss={np.mean(running_train_loss):.4f}"
            )

        mean_train_loss = np.mean(running_train_loss)
        acc_train = np.mean(np.array(all_train_preds) == np.array(all_train_targets))
        prec_train = precision_score(all_train_targets, all_train_preds, average="weighted")
        rec_train = recall_score(all_train_targets, all_train_preds, average="weighted")
        f1_train = f1_score(all_train_targets, all_train_preds, average="weighted")

        train_loss.append(mean_train_loss)
        train_acc.append(acc_train)
        train_precision.append(prec_train)
        train_recall.append(rec_train)
        train_f1.append(f1_train)

        model.eval()
        with torch.no_grad():
            running_val_loss = []
            all_val_preds, all_val_targets = [], []
            for x, targets in val_loader:
                x = x.to(device)
                targets = targets.to(device)

                pred = model(x)
                loss = loss_model(pred, targets)

                running_val_loss.append(loss.item())

                preds = pred.argmax(dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())

            mean_val_loss = np.mean(running_val_loss)
            acc_val = np.mean(np.array(all_val_preds) == np.array(all_val_targets))
            prec_val = precision_score(all_val_targets, all_val_preds, average="weighted", zero_division=0)
            rec_val = recall_score(all_val_targets, all_val_preds, average="weighted", zero_division=0)
            f1_val = f1_score(all_val_targets, all_val_preds, average="weighted", zero_division=0)

            lr_scheduler.step(metrics=mean_val_loss)
            lr = lr_scheduler.get_last_lr()
            list_.append(lr)

            val_loss.append(mean_val_loss)
            val_acc.append(acc_val)
            val_precision.append(prec_val)
            val_recall.append(rec_val)
            val_f1.append(f1_val)

        print(
            f"Epoch {epoch+1}/{epochs}; "
            f"train_loss={mean_train_loss:.4f}; train_acc={acc_train:.4f}; "
            f"train_prec={prec_train:.4f}; train_rec={rec_train:.4f}; train_f1={f1_train:.4f}; "
            f"val_loss={mean_val_loss:.4f}; val_acc={acc_val:.4f}; "
            f"val_prec={prec_val:.4f}; val_rec={rec_val:.4f}; val_f1={f1_val:.4f}"
        )

    model.eval()
    with torch.no_grad():
        running_test_loss = []
        all_test_preds, all_test_targets = [], []
        for x, targets in test_loader:
            x = x.to(device)
            targets = targets.to(device)

            pred = model(x)
            loss = loss_model(pred, targets)

            running_test_loss.append(loss.item())

            preds = pred.argmax(dim=1)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_targets.extend(targets.cpu().numpy())

        mean_test_loss = np.mean(running_test_loss)
        acc_test = np.mean(np.array(all_test_preds) == np.array(all_test_targets))
        prec_test = precision_score(all_test_targets, all_test_preds, average="weighted")
        rec_test = recall_score(all_test_targets, all_test_preds, average="weighted")
        f1_test = f1_score(all_test_targets, all_test_preds, average="weighted")

    print(
        f"test_loss={mean_test_loss:.4f}; test_acc={acc_test:.4f}; "
        f"test_prec={prec_test:.4f}; test_rec={rec_test:.4f}; test_f1={f1_test:.4f}"
    )

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['train', 'val'])
    plt.show()