import torch
from nn import RadNet
from data import HDF5Dataset, collate_fn
import sys
import numpy as np
import argparse
import glob
from functools import partial
import h5py
import os
import pickle


# Setup parser
parser = argparse.ArgumentParser(description="Arguments for RADNET")
parser.add_argument("--epochs", type=int, default=10000, help="number of epochs")
parser.add_argument(
    "--rcut", type=float, default=3.5, help="Cut off radius (in Angstrom)"
)
parser.add_argument(
    "--split", type=float, default=0.8, help="training percentage in split"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="batch size during training/validation"
)
parser.add_argument(
    "--n_outputs", type=int, default=3, help="number of outputs in neural network"
)
parser.add_argument("--filename", type=str, default=None, help="HDF5 file to read from")
parser.add_argument(
    "--max_neighbors",
    type=int,
    default=500,
    help="max number of neighbors when constructing images",
)
parser.add_argument(
    "--image_shape",
    type=int,
    nargs="+",
    default=(15, 15, 15),
    help="image sizes used to represent chemical environments",
)
parser.add_argument(
    "--sigma", type=float, default=1.0, help="sigma value used for the gaussians"
)
parser.add_argument(
    "--learning_rate", type=float, default=1e-4, help="learning rate used in training"
)
parser.add_argument(
    "--filter", type=str, default="erfc", help="filter used to blur images"
)
parser.add_argument(
    "--sample_frac",
    type=float,
    default=1,
    help="Proportion of kept data, for fast testing",
)
parser.add_argument(
    "--output_files",
    type=int,
    default=1,
    help="If 1, write the output files every epoch",
)
parser.add_argument(
    "--idx_savepath",
    type=str,
    default=None,
    help="Path to save the training/validation indices.",
)
parser.add_argument(
    "--patience",
    type=int,
    default=50,
    help="Number of consecutive epochs without val loss improvement before reducing lr.",
)
parser.add_argument(
    "--factor",
    type=float,
    default=0.9,
    help="Learning rate reduction factor",
)
parser.add_argument(
    "--stopping_patience",
    type=int,
    default=500,
    help="Number of consecutive epochs without val loss improvement before stopping training.",
)
parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay.")
parser.add_argument(
    "--augmentation", action="store_true", help="Activates data augmentation"
)

# Execute parse_args()
args = parser.parse_args()

# Initial checks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)
filename = args.filename
if filename is None:
    print("You must provide a file to train with...exiting.")
    exit()


# Set some values
batch_size = args.batch_size
split_pct = args.split
dataset = HDF5Dataset(
    filename, sample_frac=args.sample_frac, augmentation=args.augmentation
)
n_training = int(len(dataset) * split_pct)
n_validation = len(dataset) - n_training

max_epochs = args.epochs
starting_epoch = 0

# Setup model and optimization
model = RadNet(
    cut_off=args.rcut / 2,
    shape=tuple(args.image_shape),
    sigma=args.sigma,
    n_outputs=args.n_outputs,
    atom_types=dataset.unique_atomic_numbers(),
    cutoff_filter=args.filter,
    device=device,
).to(device)

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    "min",
    factor=args.factor,
    patience=args.patience,
    min_lr=3e-6,
    threshold=1e-8,
)


# Define loops
def train_loop():
    model.train()
    losses = []
    for i, batch in enumerate(training_loader):
        optimizer.zero_grad()
        preds = model(
            batch["coordinates"].to(device),
            batch["atomic_numbers"].to(device),
            batch["neighbors"].to(device),
            batch["use_neighbors"].to(device),
            batch["indices"].to(device),
        )
        trues = batch["target"].to(device)
        loss = loss_fn(preds, trues)
        losses.append(loss.detach().cpu().numpy())
        loss.backward()
        optimizer.step()
        print(f"---- batch: {i} | loss: {loss.item()} ----")
    return np.mean(losses)


def validation_loop(epoch_num):
    model.eval()
    losses = []
    all_trues = []
    all_preds = []
    for i, batch in enumerate(validation_loader):
        preds = model(
            batch["coordinates"].to(device),
            batch["atomic_numbers"].to(device),
            batch["neighbors"].to(device),
            batch["use_neighbors"].to(device),
            batch["indices"].to(device),
        )
        trues = batch["target"].to(device)
        loss = loss_fn(preds, trues)
        losses.append(loss.detach().cpu().numpy())
        all_trues.append(trues.detach().cpu())
        all_preds.append(preds.detach().cpu())
    all_trues = torch.cat(all_trues).numpy()
    all_preds = torch.cat(all_preds).numpy()

    if args.output_files:
        factor = dataset.maxs - dataset.mins
        f = open(f"tvp_epoch_{epoch_num}.dat", "w")
        for ts, ps in zip(all_trues, all_preds):
            for i, (t, p) in enumerate(zip(ts, ps)):
                f.write(
                    f"{t * factor[i] + dataset.mins[i]} {p * factor[i] + dataset.mins[i]} "
                )
            f.write("\n")

    return np.mean(losses)


def checkpoint(epoch_num, best_loss, stopping_counter, name="ckpt.torch"):
    torch.save(
        {
            "epoch": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "scheduler": scheduler.state_dict(),
            "stopping_counter": stopping_counter,
        },
        name,
    )


best_loss = np.inf
stopping_counter = 0

# Restart or not
checkpoint_files = glob.glob("ckpt*.torch")
if checkpoint_files:
    print("Found checkpoint file, continuing training")
    checkpoint_data = torch.load(checkpoint_files[0], map_location=device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint_data["scheduler"])
    starting_epoch = checkpoint_data["epoch"]
    best_loss = checkpoint_data["best_loss"]
    stopping_counter = checkpoint_data["stopping_counter"]

    with open(args.idx_savepath, "rb") as f:
        idx_dict = pickle.load(f)
    training = torch.utils.data.Subset(dataset, idx_dict["train_idx"])
    validation = torch.utils.data.Subset(dataset, idx_dict["val_idx"])
else:
    shuffled_idx = torch.randperm(len(dataset)).tolist()
    training = torch.utils.data.Subset(dataset, shuffled_idx[:n_training])
    validation = torch.utils.data.Subset(dataset, shuffled_idx[-n_validation:])

    with open(args.idx_savepath, "wb") as f:
        save = {
            "train_idx": shuffled_idx[:n_training],
            "val_idx": shuffled_idx[-n_validation:],
        }
        pickle.dump(save, f)

    print("checkpoint not found, training from scratch")


# Define loaders
training_loader = torch.utils.data.DataLoader(
    training,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=partial(
        collate_fn, cut_off=args.rcut / 2, max_neighbors=args.max_neighbors
    ),
    num_workers=0,
    pin_memory=True if device.type == "cuda" else False,
    drop_last=True,
)
validation_loader = torch.utils.data.DataLoader(
    validation,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=partial(
        collate_fn, cut_off=args.rcut / 2, max_neighbors=args.max_neighbors
    ),
    num_workers=0,
    pin_memory=True if device.type == "cuda" else False,
    drop_last=False,
)


# Training
for epoch_num in range(starting_epoch, max_epochs):
    avg_train_loss = train_loop()
    avg_validation_loss = validation_loop(epoch_num)

    scheduler.step(avg_validation_loss)

    if avg_validation_loss < best_loss:
        print("--- Better loss found, checkpointing to best.torch")
        best_loss = avg_validation_loss
        stopping_counter = 0
        checkpoint(epoch_num, best_loss, stopping_counter, name="best.torch")
    else:
        stopping_counter += 1

    checkpoint(epoch_num, best_loss, stopping_counter, name=f"ckpt_{epoch_num}.torch")

    try:
        os.remove(f"ckpt_{epoch_num - 1}.torch")
    except:
        pass

    print(
        f"-- epoch: {epoch_num} | train_loss: {avg_train_loss} | validation_loss: {avg_validation_loss}"
    )
    print(
        f"       learning rate: {optimizer.param_groups[0]['lr']} | stopping_counter: {stopping_counter}"
    )

    if stopping_counter == args.stopping_patience:
        break

print("Training complete.")
