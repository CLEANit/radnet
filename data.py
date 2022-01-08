import torch
import h5py
import numpy as np
from ase import Atoms, neighborlist 

def _make_float32(samples):
    for k, v in samples.items():
        if v.dtype == torch.float64:
            samples[k] = v.float()
    return samples

def flatten(samples):
    cut_off = 1.852 / 2
    max_neighbors = 500

    return_dict = {}
    for k in samples[0]:
        return_dict[k] = []
    return_dict['indices'] = []
    return_dict['cell'] = []
    return_dict['neighbors'] = []
    return_dict['use_neighbors'] = []
    for i, sample in enumerate(samples):
        atomic_numbers = sample['atomic_numbers'].tolist()
        coordinates = sample['coordinates'].tolist()
        cell = sample['cell'].tolist()

        # construct neighbor list on the fly
        nl = neighborlist.NeighborList([cut_off] * len(atomic_numbers), 
                    bothways=True,
                    self_interaction=True,
                    sorted=False)
        atoms = Atoms(numbers=atomic_numbers, cell=cell, positions=coordinates, pbc=True)
        nl.update(atoms)
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
            return_dict['use_neighbors'].append(use_neighbors.tolist())
            return_dict['neighbors'].append(relative_pos.tolist())
        return_dict['atomic_numbers'] += atomic_numbers
        return_dict['coordinates'] += coordinates
        return_dict['indices'] += [i] * len(atomic_numbers)
        return_dict['cell'] += [cell] * len(atomic_numbers)
        for k, v in sample.items():
            if k != 'atomic_numbers' and k != 'coordinates' and k != 'indices' and k != 'cell':
                return_dict[k].append(v.tolist())

    for k, v in return_dict.items():
        return_dict[k] = torch.tensor(v)
    return return_dict

def collate_fn(samples):
    samples = flatten(samples)
    return _make_float32(samples)


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, normalize=True, cut_off=3.0, max_neighbors=50):
        self.filename = filename
        self.cut_off = cut_off
        self.max_neighbors = max_neighbors
        self._load_data()

        if normalize:
            self._normalize()

    def _load_data(self):
        """
        This loads everything into memory, need to do something else when using a bigger dataset
        """
        f = h5py.File(self.filename, 'r')
        self.data = []
        print('Loading data...')
        for struct_name, struct_vals in f.items():
            for i, pos in enumerate(struct_vals['coordinates'][:]):
                data_point = {
                    'atomic_numbers': struct_vals['atomic_numbers'][:],
                    'cell': struct_vals['cell'][:],
                    'coordinates': pos,
                    'target': struct_vals['target'][i],
                }
                # print('Done with:', struct_name, 'coordinate:', i)
                self.data.append(data_point)

    def _normalize(self):
        vals = []
        for elem in self.data:
            vals.append(elem['target'])

        vals = np.array(vals)
        mins = vals.min(0)
        maxs = vals.max(0)
        print('INFO from normalization:')
        print('mins:', mins)
        print('maxs:', maxs)

        for i, elem in enumerate(self.data):
            self.data[i]['target'] = (elem['target'] - mins) / (maxs - mins)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
