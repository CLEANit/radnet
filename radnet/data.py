import torch
import h5py
import numpy as np
from ase import Atoms, neighborlist
from ase.cell import Cell
import ase.neighborlist
from ase.units import Bohr
from radnet.utils import (
    _make_float32,
    target_to_tensor,
    tensor_to_target,
    generate_random_3D_rotation_matrix,
)
from copy import deepcopy


def flatten(samples, cut_off, max_neighbors):
    return_dict = {}
    for k in samples[0]:
        return_dict[k] = []
    return_dict["indices"] = []
    return_dict["neighbors"] = []
    return_dict["use_neighbors"] = []
    for i, sample in enumerate(samples):
        atomic_numbers = sample["atomic_numbers"].tolist()
        coordinates = sample["coordinates"].tolist()
        cell = sample["cell"].tolist()

        # construct neighbor list on the fly
        nl = neighborlist.NeighborList(
            [cut_off] * len(atomic_numbers),
            skin=0,
            bothways=True,
            self_interaction=True,
            sorted=False,
            primitive=ase.neighborlist.NewPrimitiveNeighborList,
        )
        atoms = Atoms(
            numbers=atomic_numbers, cell=cell, positions=coordinates, pbc=True
        )
        nl.update(atoms)
        neighbors_list, use_neighbors_list = [], []
        for j, atom in enumerate(atoms):
            indices, offsets = nl.get_neighbors(j)
            relative_pos = np.zeros((max_neighbors, 3))
            use_neighbors = np.zeros(max_neighbors)
            total = 0
            for k, offset in zip(indices, offsets):
                if total < max_neighbors:
                    relative_pos[total] = atoms.positions[k] + offset @ atoms.get_cell()
                    use_neighbors[total] = 1
                total += 1
            return_dict["use_neighbors"].append(use_neighbors.tolist())
            return_dict["neighbors"].append(relative_pos.tolist())
        return_dict["atomic_numbers"] += atomic_numbers
        return_dict["coordinates"] += coordinates
        return_dict["indices"] += [i] * len(atomic_numbers)
        for k, v in sample.items():
            if (
                k != "atomic_numbers"
                and k != "coordinates"
                and k != "indices"
                and k != "cell"
            ):
                return_dict[k].append(v.tolist())

    for k, v in return_dict.items():
        return_dict[k] = torch.tensor(v)
    return return_dict


def collate_fn(samples, cut_off, max_neighbors):
    samples = flatten(samples, cut_off, max_neighbors)
    return _make_float32(samples)


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filename,
        normalize=True,
        normalize_mode="data",
        augmentation_mode=None,
        sample_frac=1,
    ):
        self.filename = filename
        self.sample_frac = sample_frac
        self._load_data()

        assert augmentation_mode in [
            None,
            "reflections",
            "rotations",
            "symmetries",
            "symmetries_6",
            "symmetries_24",
        ]
        if augmentation_mode in ["symmetries_6", "symmetries_24"]:
            self.n_symmetries = int(augmentation_mode.split("_")[-1])
            self.augmentation_mode = "symmetries"
        else:
            self.n_symmetries = (
                24 if self.n_outputs == 3 else 6
            )  # Only used for symmetries augmentation
            self.augmentation_mode = augmentation_mode

        if self.augmentation_mode == "symmetries":
            # Needs to be generalized to other materials
            from radnet.utils import get_symmetries_array

            self.symmetries_array = get_symmetries_array(self.n_symmetries)

        self.normalize = normalize
        if self.normalize:
            assert normalize_mode in ["data", "file"]
            self._initial_normalization(normalize_mode)

    def _load_data(self):
        """
        This loads everything into memory, need to do something else when using a bigger dataset
        """
        f = h5py.File(self.filename, "r")
        self.data = []
        print("Loading data...")
        self.ams = []
        for struct_name, struct_vals in f.items():
            max_samples = int(self.sample_frac * struct_vals["coordinates"].shape[0])
            for i, pos in enumerate(struct_vals["coordinates"][:]):
                data_point = {
                    "atomic_numbers": struct_vals["atomic_numbers"][:],
                    "cell": struct_vals["cell"][:],
                    "coordinates": pos,
                    "target": struct_vals["target"][i],
                }
                try:
                    data_point["polarization_phases"] = struct_vals[
                        "polarization_phases"
                    ][i]
                except:
                    pass

                self.data.append(data_point)
                self.ams += struct_vals["atomic_numbers"][:].tolist()
                if i >= max_samples:
                    break
        self.n_outputs = self.data[0]["target"].shape[0]
        if self.n_outputs not in [3, 6]:
            raise RuntimeError(
                f"The output target shape {self.n_outputs} is not recognized."
            )
        if self.n_outputs == 6:
            for datapoint in self.data:
                datapoint["target_tensor"] = target_to_tensor(datapoint["target"])

        # To implement differently for varying cell sizes
        self.cell = Cell(self.data[0]["cell"])
        self.au_cell = self.cell / Bohr
        self.au_volume = self.cell.volume / (Bohr**3)

        max_cell_dim = np.max(np.sum(np.abs(self.cell), axis=0))
        self.bias_cell_lims = (-max_cell_dim, max_cell_dim)

    def _initial_normalization(self, normalize_mode):
        if normalize_mode == "data":
            vals = []
            for elem in self.data:
                vals.append(elem["target"])

            vals = np.array(vals)
            mins = vals.min(0)
            maxs = vals.max(0)

            for i, (min_val, max_val) in enumerate(zip(mins, maxs)):
                if min_val == max_val:
                    mins[i], maxs[i] = mins[i] - 0.01, maxs[i] + 0.01

            if (
                self.augmentation_mode in ["reflections", "rotations"]
                and self.n_outputs == 3
            ):
                maxval = max(np.absolute(maxs).max(), np.absolute(mins).max())
                maxs = np.array([maxval, maxval, maxval])
                mins = np.array([-maxval, -maxval, -maxval])
            elif self.augmentation_mode == "symmetries" or self.n_outputs == 6:
                values = []
                for elem in self.data:
                    elem = deepcopy(elem)
                    elem = self._augment_data(elem)
                    values.append(elem["target"])
                values = np.array(values)
                mins = values.min(0)
                maxs = values.max(0)
            else:
                pass

            # Check for 0 norm
            idx = np.where(mins == maxs)[0]
            maxs[idx] += maxs[idx] / 2
            mins[idx] -= mins[idx] / 2

            print("INFO from normalization:")
            print("mins:", mins)
            print("maxs:", maxs)

            print('writing to file "normalization_info.dat"')
            f = open("normalization_info.dat", "w")
            f.write("Mins:\n")
            for minelem in mins:
                f.write(str(minelem) + "\t")
            f.write("\n")
            f.write("Maxs:\n")
            for maxelem in maxs:
                f.write(str(maxelem) + "\t")
            f.write("\n")
            f.close()

        elif normalize_mode == "file":
            f = open("normalization_info.dat", "r")
            txt = f.read().split("\n")
            f.close()
            mins = np.array(txt[1].split("\t")[:-1], dtype=float)
            maxs = np.array(txt[3].split("\t")[:-1], dtype=float)
            print("mins:", mins)
            print("maxs:", maxs)

        self.mins = mins
        self.maxs = maxs

    def _normalize_data(self, datapoint):
        datapoint["target"] = (datapoint["target"] - self.mins) / (
            self.maxs - self.mins
        )
        return datapoint

    def unique_atomic_numbers(self):
        return list(set(self.ams))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datapoint = deepcopy(self.data[idx])
        if self.augmentation_mode:
            datapoint = self._augment_data(datapoint)
        if self.normalize:
            datapoint = self._normalize_data(datapoint)
        return datapoint

    def _get_random_3D_rotation_matrix(self):
        if self.augmentation_mode == "rotations":
            M = generate_random_3D_rotation_matrix()

        elif self.augmentation_mode == "reflections":
            # Version reflections in all directions
            diag = (np.random.rand(3) - 0.5 < 0).astype(int) * 2 - 1
            M = np.eye(3) * diag
        return M

    def _augment_data(self, datapoint):
        scaled_positions = self.cell.scaled_positions(datapoint["coordinates"])

        if self.augmentation_mode in ["reflections", "rotations"]:
            rotation_matrix = self._get_random_3D_rotation_matrix()
            rotated_cell = self.cell.array @ rotation_matrix.T
            datapoint["cell"] = rotated_cell
            datapoint["coordinates"] = scaled_positions @ rotated_cell

            if self.n_outputs == 3:
                datapoint["target"] = rotation_matrix @ datapoint["target"]
            elif self.n_outputs == 6:
                datapoint["target"] = tensor_to_target(
                    rotation_matrix @ datapoint["target_tensor"] @ rotation_matrix.T
                )

        elif self.augmentation_mode == "symmetries":
            # df = np.random.randint(self.symmetries_array.shape[0])
            symmetry_matrix = self.symmetries_array[
                np.random.randint(self.symmetries_array.shape[0])
            ]
            # print(symmetry_matrix)
            # print("before: ", datapoint["coordinates"][14])

            new_scaled_positions = 0
            for dim in range(3):
                new_scaled_positions += (
                    np.tile(
                        symmetry_matrix[:, dim].reshape(3, -1),
                        scaled_positions.shape[0],
                    )
                    * scaled_positions[:, dim]
                ).T
            datapoint["coordinates"] = new_scaled_positions @ self.cell.array
            # print("after: ", datapoint["coordinates"][14])

            if self.n_outputs == 3:
                phases = datapoint["polarization_phases"]
                phases = (symmetry_matrix * phases).sum(1) % 2
                phases = np.broadcast_to(phases, (3, 3)).T
                datapoint["target"] = (phases * self.au_cell).sum(0) / self.au_volume
                # print(datapoint["target"])
            elif self.n_outputs == 6:
                datapoint["target"] = tensor_to_target(
                    symmetry_matrix @ datapoint["target_tensor"] @ symmetry_matrix.T
                )
            # print(datapoint["target"])
        return datapoint
