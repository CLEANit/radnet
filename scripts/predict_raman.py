import torch
import h5py
import os
from radnet.nn import RadNet
from radnet.data import HDF5Dataset, collate_fn
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
    parser.add_argument(
        "prediction",
        choices=["pol", "effch", "dielectric", "suscept_deriv", "raman_tensor"],
        help="Type of prediction.",
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
        "--sigma", type=float, default=0.5, help="sigma value used for the gaussians"
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
        help="Biased models to learn rotations.",
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
        "--phonon_file",
        type=str,
        default=None,
        help="h5 file containing the dft eigenmodes for raman tensor calculation.",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Whether to save the results for further analysis.",
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
        biased_filters=args.biased_model,
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
    derivatives = torch_derivative(polarization, coordinates)
    effective_charges = torch.transpose(derivatives, 0, 1)
    effective_charges *= atoms.cell.volume / Bohr**3
    if asr:
        # Mean should be 0
        effective_charges -= torch.mean(effective_charges, dim=0)
    return effective_charges


def get_dielectric_tensor(batch, model, device, mins, maxs):
    coordinates = batch["coordinates"].to(device).requires_grad_()
    dielectric = model(
        coordinates,
        batch["atomic_numbers"].to(device),
        batch["neighbors"].to(device),
        batch["use_neighbors"].to(device),
        batch["indices"].to(device),
    )
    dielectric = dielectric[0] * (maxs - mins) + mins
    dielectric = reshape_dielectric_tensor(dielectric)
    return dielectric, coordinates


def reshape_dielectric_tensor(dielectric):
    out = torch.empty((3, 3))
    for i in range(3):
        for j in range(3):
            if (i == 0) or (j == 0):
                out[i, j] = dielectric[i + j]
            else:
                out[i, j] = dielectric[i + j + 1]
    return out


def get_suscept_deriv(dielectric, coordinates, atoms, asr=True):
    dielectric_deriv = torch_derivative(dielectric, coordinates)
    suscept_deriv = (4 * np.pi) ** (-1) * dielectric_deriv
    suscept_deriv = (
        suscept_deriv.transpose(0, 1).transpose(1, 2).reshape(len(atoms), 3, 3, 3)
    )
    if asr:
        suscept_deriv -= torch.mean(suscept_deriv, dim=0)
    return suscept_deriv


def read_eigenmodes(phonon_file):
    assert (
        args.phonon_file is not None
    ), "A phonon file should be given for this calculation."
    phonon_file = phonon_file if phonon_file.endswith(".h5") else phonon_file + ".h5"
    import h5py

    with h5py.File(phonon_file) as f:
        eigenmodes = f["eigenmodes"][:]
    return eigenmodes


def get_raman_tensor(suscept_deriv, eigenmodes, atoms):
    nat = int(eigenmodes.shape[0] / 3)
    assert nat == suscept_deriv.shape[0]

    suscept_deriv = suscept_deriv.detach().cpu().numpy()
    raman_tensors = np.zeros((nat * 3, 3, 3))
    for l in range(nat * 3):
        tensor = np.zeros((3, 3))
        for m in range(nat):
            for alpha in range(3):
                tensor += suscept_deriv[m, alpha] * eigenmodes[l, m, alpha]
        raman_tensors[l] = tensor

    raman_tensors *= np.sqrt(atoms.cell.volume / Bohr**3)
    return raman_tensors


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
    if args.prediction in ["pol", "effch"]:
        assert model.n_outputs == 3

        for batch in prediction_loader:
            polarization, coordinates = get_polarization(
                batch, model, device, mins, maxs
            )
            if args.prediction == "effch":
                effective_charges = get_effective_charges(
                    polarization, coordinates, atoms, asr=True
                )

        if args.prediction == "pol":
            print("Polarization values: ")
            print(polarization)
            if args.save_results:
                np.save("polarization.npy", polarization.detach().cpu().numpy())

        elif args.prediction == "effch":
            print("Born effective charges:")
            print(effective_charges)
            if args.save_results:
                np.save(
                    "effective_charges.npy", effective_charges.detach().cpu().numpy()
                )

    elif args.prediction in ["dielectric", "suscept_deriv", "raman_tensor"]:
        assert model.n_outputs == 6

        for batch in prediction_loader:
            dielectric, coordinates = get_dielectric_tensor(
                batch, model, device, mins, maxs
            )
            if args.prediction in ["suscept_deriv", "raman_tensor"]:
                suscept_deriv = get_suscept_deriv(dielectric, coordinates, atoms)

                if args.prediction == "raman_tensor":
                    eigenmodes = read_eigenmodes(args.phonon_file)
                    raman_tensor = get_raman_tensor(suscept_deriv, eigenmodes, atoms)

        if args.prediction == "dielectric":
            print("Dielectric tensor:")
            print(dielectric)
            if args.save_results:
                np.save("dielectric.npy", dielectric.detach().cpu().numpy())

        elif args.prediction == "suscept_deriv":
            print("Electric susceptibility derivatives:")
            print(suscept_deriv)
            if args.save_results:
                np.save("suscept_deriv.npy", suscept_deriv.detach().cpu().numpy())

        elif args.prediction == "raman_tensor":
            print("Raman tensor:")
            print(raman_tensor)
            if args.save_results:
                np.save("raman_tensor.npy", raman_tensor)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
