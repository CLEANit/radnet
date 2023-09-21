import torch
import h5py
import os
from nn import RadNet
from data import HDF5Dataset, collate_fn
from functools import partial
import argparse
import pickle
import numpy as np
from ase.io import read
from ase.units import Bohr


def build_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for RadNet prediction of properties for raman spectra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pos_file",
        type=str,
        help="Path to the positions file.",
    )
    parser.add_argument("prediction", choices=["effch"], help="Type of prediction.")
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
        "--sigma", type=float, default=0.5, help="sigma value used for the gaussians"
    )
    parser.add_argument(
        "--filter", type=str, default="erfc", help="filter used to blur images"
    )
    parser.add_argument(
        "--max_neighbors",
        type=int,
        default=500,
        help="max number of neighbors when constructing images",
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
    return parser


def torch_derivative(fx, x, create_graph=False):
    dfdx = []
    flat_fx = fx.reshape(-1)
    for i in range(len(flat_fx)):
        (grad_x,) = torch.autograd.grad(
            flat_fx[i],
            x,
            torch.ones_like(flat_fx[i]),
            retain_graph=True,
            create_graph=create_graph,
        )
        dfdx.append(grad_x)
    return torch.stack(dfdx)


def create_dataset(args, atoms):
    with h5py.File("temp.h5", "w") as temp:
        group = temp.create_group("struct")
        group.create_dataset("atomic_numbers", data=atoms.get_atomic_numbers())
        group.create_dataset("cell", data=atoms.cell)
        group.create_dataset(
            "coordinates", data=np.expand_dims(atoms.get_positions(), axis=0)
        )
        group.create_dataset(
            "target", data=np.expand_dims(np.zeros(args.n_outputs), axis=0)
        )

    dataset = HDF5Dataset(
        "temp.h5",
        normalize=True,
        normalize_mode="file",
        augmentation=False,
    )
    os.remove("temp.h5")
    return dataset


def create_dataloader(args, dataset, device):
    prediction_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=partial(
            collate_fn, cut_off=args.rcut / 2, max_neighbors=args.max_neighbors
        ),
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=False,
    )
    return prediction_loader


def get_model(args, dataset, device):
    model = RadNet(
        cut_off=args.rcut / 2,
        shape=tuple(args.image_shape),
        sigma=args.sigma,
        n_outputs=args.n_outputs,
        atom_types=dataset.unique_atomic_numbers(),
        cutoff_filter=args.filter,
        device=device,
    ).to(device)
    return model


def get_polarization(batch, model, device, mins, maxs):
    coordinates = batch["coordinates"].to(device).requires_grad_()
    polarization = model(
        coordinates,
        batch["atomic_numbers"].to(device),
        batch["neighbors"].to(device),
        batch["use_neighbors"].to(device),
        batch["indices"].to(device),
    )
    polarization = polarization * (maxs - mins) + mins
    return polarization, coordinates


def get_effective_charges(polarization, coordinates, atoms, asr=True):
    # effective_charges = torch.empty((coordinates.shape[0], 3, 3))
    derivatives = torch_derivative(polarization, coordinates)
    # for i in range(len(derivatives)):
    #    for j in range(len(derivatives[i])):
    #        effective_charges[j, i] = derivatives[i][j]
    effective_charges = torch.transpose(derivatives, 0, 1)
    effective_charges *= atoms.cell.volume / Bohr**3
    if asr:
        # Mean should be 0
        effective_charges -= torch.mean(effective_charges, dim=0)
    return effective_charges


def get_loto_splitting():
    pass


def main(args):
    device = torch.device(args.device)
    atoms = read(args.pos_file)
    dataset = create_dataset(args, atoms)
    prediction_loader = create_dataloader(args, dataset, device)

    model = get_model(args, dataset, device)

    model.load_state_dict(
        torch.load(args.saved_model_path, map_location=device)["model_state_dict"]
    )
    model.eval()

    mins = torch.tensor(dataset.mins).to(device)
    maxs = torch.tensor(dataset.maxs).to(device)

    torch.set_printoptions(precision=8)
    if args.prediction in ["pol", "effch", "loto"]:
        assert model.n_outputs == 3

        for batch in prediction_loader:
            polarization, coordinates = get_polarization(
                batch, model, device, mins, maxs
            )
            if args.prediction in ["effch", "loto"]:
                effective_charges = get_effective_charges(
                    polarization, coordinates, atoms, asr=True
                )
            if args.prediction in ["loto"]:
                loto_splitting = get_loto_splitting()

        if args.prediction == "pol":
            print("Polarization values: ", polarization)
        elif args.prediction == "effch":
            print("Born effective charges:")
            print(effective_charges)
        elif args.prediction == "loto":
            pass


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
