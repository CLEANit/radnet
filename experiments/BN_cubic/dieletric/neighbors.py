import h5py
import ase
import ase.neighborlist

f = h5py.File("BN_cubic_data.h5")
at = ase.Atoms(f['BN']["atomic_numbers"], cell=f['BN']["cell"], positions = f['BN']["coordinates"][0])

for cutoff in [0.4, 0.5]:
    nl = ase.neighborlist.NeighborList([cutoff]*16, bothways=True, self_interaction=True, sorted=False,
            primitive=ase.neighborlist.NewPrimitiveNeighborList)
    nl.update(at)
    print(cutoff)
    print(nl.get_neighbors(0))
