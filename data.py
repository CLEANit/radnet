import torch
import h5py
import numpy as np
from ase import Atoms, neighborlist
from ase.cell import Cell
import ase.neighborlist


def _make_float32(samples):
    for k, v in samples.items():
        if v.dtype == torch.float64:
            samples[k] = v.float()
    return samples


def target_to_tensor(target):
    return np.array(
        [
            [target[0], target[1], target[2]],
            [target[1], target[3], target[4]],
            [target[2], target[4], target[5]],
        ]
    )


def tensor_to_target(tensor):
    return np.array(
        [
            tensor[0, 0],
            tensor[0, 1],
            tensor[0, 2],
            tensor[1, 1],
            tensor[1, 2],
            tensor[2, 2],
        ]
    )


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

        assert augmentation_mode in [None, "reflections", "rotations"]
        self.augmentation_mode = augmentation_mode

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
        example_cell = self.data[0]["cell"]

        max_cell_dim = np.max(np.sum(np.abs(example_cell), axis=0))
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

            if self.augmentation_mode:
                if self.n_outputs == 3:
                    maxval = max(np.absolute(maxs).max(), np.absolute(mins).max())
                    maxs = np.array([maxval, maxval, maxval])
                    mins = np.array([-maxval, -maxval, -maxval])
                elif self.n_outputs == 6:
                    values = []
                    for elem in self.data:
                        elem = self._augment_data(elem)
                        values.append(elem["target"])
                    values = np.array(values)
                    mins = values.min(0)
                    maxs = values.max(0)
                    pass

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
        datapoint = self.data[idx].copy()
        if self.augmentation_mode:
            datapoint = self._augment_data(datapoint)
        if self.normalize:
            datapoint = self._normalize_data(datapoint)
        return datapoint

    def _get_random_3D_rotation_matrix(self):
        r"Algorithm from James Avro, https://doi.org/10.1016/B978-0-08-050755-2.50034-8"

        if self.augmentation_mode == "rotations":

            def generate_random_z_axis_rotation():
                R = np.eye(3)
                x1 = np.random.rand()
                R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
                R[0, 1] = -np.sin(2 * np.pi * x1)
                R[1, 0] = np.sin(2 * np.pi * x1)
                return R

            x2 = 2 * np.pi * np.random.rand()
            x3 = np.random.rand()

            R = generate_random_z_axis_rotation()
            v = np.array(
                [np.cos(x2) * np.sqrt(x3), np.sin(x2) * np.sqrt(x3), np.sqrt(1 - x3)]
            )
            H = np.eye(3) - (2 * np.outer(v, v))
            M = -(H @ R)

        elif self.augmentation_mode == "reflections":
            # Version reflections in all directions
            diag = (np.random.rand(3) - 0.5 < 0).astype(int) * 2 - 1
            M = np.eye(3) * diag
        return M

    def _augment_data(self, datapoint):
        cell = Cell(datapoint["cell"])
        scaled_positions = cell.scaled_positions(datapoint["coordinates"])
        rotation_matrix = self._get_random_3D_rotation_matrix()
        rotated_cell = cell.array @ rotation_matrix.T
        datapoint["cell"] = rotated_cell
        datapoint["coordinates"] = scaled_positions @ rotated_cell

        if self.n_outputs == 3:
            datapoint["target"] = rotation_matrix @ datapoint["target"]
        elif self.n_outputs == 6:
            datapoint["target"] = tensor_to_target(
                rotation_matrix @ datapoint["target_tensor"] @ rotation_matrix.T
            )
        return datapoint
