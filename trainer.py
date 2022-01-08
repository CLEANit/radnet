import torch
from nn import RadNet
from data import HDF5Dataset, collate_fn
import sys
import numpy as np
import argparse 

filename = sys.argv[1]

batch_size = 32
split_pct = 0.8
dataset = HDF5Dataset(filename)
n_training = int(len(dataset) * split_pct)
n_validation = len(dataset) - n_training
training, validation = torch.utils.data.random_split(dataset, (n_training, n_validation))
training_loader = torch.utils.data.DataLoader(
    training, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=False
    )
validation_loader = torch.utils.data.DataLoader(
    validation, 
    batch_size=batch_size, 
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=False
    )

max_epochs = 10000
starting_epoch = 0
model = RadNet()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

def train_loop():
    model.train()
    losses = []
    for i, batch in enumerate(training_loader):
        optimizer.zero_grad()
        preds = model(batch['coordinates'], batch['atomic_numbers'], batch['neighbors'], batch['use_neighbors'], batch['cell'], batch['indices'])
        trues = batch['target']
        loss = loss_fn(preds, trues)
        losses.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
        print(f'---- batch: {i} | loss: {loss.item()} ----')
    return np.mean(losses)


def validation_loop():
    model.eval()
    losses = []
    for i, batch in enumerate(validation_loader):
        preds = model(batch['coordinates'], batch['atomic_numbers'], batch['neighbors'], batch['use_neighbors'], batch['cell'], batch['indices'])
        trues = batch['target']
        loss = loss_fn(preds, trues)
        losses.append(loss.detach().numpy())
    return np.mean(losses)


for epoch_num in range(starting_epoch, max_epochs):
    avg_train_loss = train_loop()
    avg_validation_loss = validation_loop()
    print(f'-- epoch: {epoch_num} | train_loss: {avg_train_loss} | validation_loss: {avg_validation_loss}')





