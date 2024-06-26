from radnet.data import HDF5Dataset, collate_fn
from radnet.nn import RadNet
from functools import partial
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import torch


parser = argparse.ArgumentParser(
    description="Arguments for inference of RadNet",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "datapath",
    type=str,
    help="Path to the dataset to evaluate.",
)
parser.add_argument(
    "--idx_filepath",
    type=str,
    default=None,
    help="""
    Path to the pickle file containing the indices
    of the data points in datapath to evaluate.
    """,
)
parser.add_argument(
    "--manual_index", type=int, default=None, help="Choose an index of the dataset."
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for the inference.",
)
parser.add_argument(
    "--rcut", type=float, default=3.5, help="Cut off radius (in Angstrom)"
)
parser.add_argument(
    "--n_outputs", type=int, default=3, help="number of outputs in neural network"
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
    "--filter", type=str, default="erfc", help="filter used to blur images"
)
parser.add_argument(
    "--input_normalization",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Normalize the english muffins before going in the model.",
)
parser.add_argument(
    "--max_neighbors",
    type=int,
    default=500,
    help="max number of neighbors when constructing images",
)
parser.add_argument(
    "--biased_model",
    action="store_true",
    default=False,
    help="Used biased models to learn rotations.",
)
parser.add_argument(
    "--saved_model_path",
    type=str,
    default="best.torch",
    help="Path to the checkpoint file to load the model from.",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    choices=("cpu", "cuda"),
    help="Device.",
)
parser.add_argument(
    "--augmentation_mode",
    default=None,
    choices=[
        None,
        "reflections",
        "rotations",
        "symmetries",
        "symmetries_6",
        "symmetries_24",
    ],
    help="Activates data augmentation",
)
parser.add_argument(
    "--detailed",
    action="store_true",
    help="If True, write detailed results",
)
args = parser.parse_args()


device = torch.device(args.device)
batch_size = args.batch_size


dataset = HDF5Dataset(
    args.datapath,
    normalize=True,
    normalize_mode="file",
    augmentation_mode=args.augmentation_mode,
)

# Choose the subset of data to evaluate if needed
if (args.idx_filepath is not None) and (args.manual_index is not None):
    raise RuntimeError("Choose max one of idx_filename or manual_index.")

if args.idx_filepath is not None:
    with open(args.idx_filepath, "rb") as f:
        indices = pickle.load(f)["val_idx"]
    inference_dataset = torch.utils.data.Subset(dataset, indices)
elif args.manual_index is not None:
    inference_dataset = torch.utils.data.Subset(dataset, [args.manual_index])
else:
    inference_dataset = dataset

inference_loader = torch.utils.data.DataLoader(
    inference_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=partial(
        collate_fn, cut_off=args.rcut / 2, max_neighbors=args.max_neighbors
    ),
    num_workers=0,
    pin_memory=True if device.type == "cuda" else False,
    drop_last=False,
)

model = RadNet(
    cut_off=args.rcut / 2,
    shape=tuple(args.image_shape),
    sigma=args.sigma,
    n_outputs=args.n_outputs,
    atom_types=dataset.unique_atomic_numbers(),
    cutoff_filter=args.filter,
    biased_filters=args.biased_model,
    bias_cell_lims=dataset.bias_cell_lims,
    device=device,
).to(device)

model.load_state_dict(
    torch.load(args.saved_model_path, map_location=device)["model_state_dict"]
)
model.eval()

mins = torch.tensor(dataset.mins).to(device)
maxs = torch.tensor(dataset.maxs).to(device)

absolute_errors = []
squared_errors = []

if args.detailed:
    all_trues, all_preds = [], []

with torch.no_grad():
    for i, batch in enumerate(tqdm(inference_loader)):
        preds = model(
            batch["coordinates"].to(device),
            batch["atomic_numbers"].to(device),
            batch["neighbors"].to(device),
            batch["use_neighbors"].to(device),
            batch["indices"].to(device),
        )
        preds = preds * (maxs - mins) + mins
        trues = batch["target"].to(device)
        trues = trues * (maxs - mins) + mins
        if args.detailed:
            all_preds.append(preds)
            all_trues.append(trues)

        absolute_errors.append(
            torch.mean(torch.abs(preds - trues), dim=-1).cpu().numpy()
        )
        squared_errors.append(torch.mean((preds - trues) ** 2, dim=-1).cpu().numpy())

mae = np.mean(np.concatenate(absolute_errors))
rmse = np.sqrt(np.mean(np.concatenate(squared_errors)))

print("Prediction metrics:")
print(f" --MAE: {mae}")
print(f" --RMSE: {rmse}")

with open("inference_results.txt", "w") as f:
    f.write(f"{mae}\t{rmse}")

if args.detailed:
    all_trues = np.concatenate([t.cpu().numpy() for t in all_trues])
    all_preds = np.concatenate([p.cpu().numpy() for p in all_preds])

    with open("detailed_results.pkl", "wb") as f:
        pickle.dump({"trues": all_trues, "preds": all_preds}, f)
