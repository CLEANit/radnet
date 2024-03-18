import argparse
import ase
import h5py
import numpy as np
from ase.io import read
from ase.units import Bohr, invcm, Ha


AMU_TO_EMU = 1.660538782e-27 / 9.10938215e-31


def build_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for calculation of the LO-TO splitting and Raman spectrum",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pos_file", type=str, help="Path to the positions file.")
    parser.add_argument(
        "dft_phonon_file",
        type=str,
        default=None,
        help="h5 file containing the dft eigenmodes for raman tensor calculation.",
    )
    parser.add_argument(
        "--charge_path",
        type=str,
        default="effective_charge.npy",
        help="Path to the file containing the effective charges.",
    )
    parser.add_argument(
        "--die_path",
        type=str,
        default="dielectric.npy",
        help="Path to the file containing the dielectric tensor.",
    )
    parser.add_argument(
        "--raman_tensor_path",
        type=str,
        default="raman_tensor.npy",
        help="Path to the file containing the raman tensor.",
    )
    parser.add_argument(
        "--dft_raman_tensor_path",
        type=str,
        default=None,
        help="Use to also compute DFT Raman intensities.",
    )
    return parser


def get_squared_loto_splittings(
    atoms: ase.Atoms, effective_charges: np.ndarray, dielectric: np.ndarray
) -> np.ndarray:
    nat = len(atoms)
    volume = atoms.cell.volume / Bohr**3

    masses = atoms.get_masses() * AMU_TO_EMU
    qvecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    squared_splittings = []
    for qvec in qvecs:
        squared_splitting = 0
        for m in range(nat):
            mass = masses[m]

            numerator, denominator = 0, 0
            for alpha in range(3):
                for beta in range(3):
                    denominator += qvec[alpha] * dielectric[alpha, beta] * qvec[beta]
                    for gamma in range(3):
                        numerator += (
                            qvec[alpha]
                            * effective_charges[m, alpha, beta]
                            * effective_charges[m, gamma, beta]
                            * qvec[gamma]
                        )
            squared_splitting += mass ** (-1) * numerator / denominator
        squared_splitting = (4 * np.pi / volume) * squared_splitting
        squared_splittings.append(squared_splitting)
    return np.array(squared_splittings)


def get_gamma_modes_idx(eigenmodes: np.ndarray) -> np.ndarray:
    idx = []
    for i, mode in enumerate(eigenmodes):
        temp = mode.reshape(int(mode.shape[0] / 2), 2, mode.shape[1])
        if np.all(np.isclose(temp[0], temp)):
            idx.append(i)
    return np.array(idx)


def get_loto_splitting(
    squared_splittings: np.ndarray, eigenmodes: np.ndarray, eigenenergies: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray):
    gamma_modes_idx = get_gamma_modes_idx(eigenmodes)
    gamma_eigenenergies = np.sort(eigenenergies[gamma_modes_idx])
    base_eigenvalue = gamma_eigenenergies[-2] * invcm / Ha

    loto = (
        (np.sqrt(base_eigenvalue**2 + squared_splittings) - base_eigenvalue)
        * Ha
        / invcm
    )
    error = loto - (gamma_eigenenergies[-1] - gamma_eigenenergies[-2])

    ml_eigenenergies = eigenenergies.copy()
    gamma_eigen = ml_eigenenergies[gamma_modes_idx]
    gamma_eigen[-1] = gamma_eigen[-2] + np.mean(loto)
    ml_eigenenergies[gamma_modes_idx] = gamma_eigen

    return loto, error, ml_eigenenergies


def get_rotation_invariants(raman_tensor: np.ndarray) -> np.ndarray:
    rotation_invariants = np.empty((raman_tensor.shape[0], 3))

    for l in range(rotation_invariants.shape[0]):
        rotation_invariants[l, 0] = (1 / 3) * (
            raman_tensor[l][0, 0] + raman_tensor[l][1, 1] + raman_tensor[l][2, 2]
        ) ** 2
        rotation_invariants[l, 1] = (1 / 2) * (
            (raman_tensor[l][0, 1] - raman_tensor[l][1, 0]) ** 2
            + (raman_tensor[l][2, 1] - raman_tensor[l][1, 2]) ** 2
            + (raman_tensor[l][0, 2] - raman_tensor[l][2, 0]) ** 2
        )
        rotation_invariants[l, 2] = (1 / 2) * (
            (raman_tensor[l][0, 1] + raman_tensor[l][1, 0]) ** 2
            + (raman_tensor[l][1, 2] + raman_tensor[l][2, 1]) ** 2
            + (raman_tensor[l][0, 2] + raman_tensor[l][2, 0]) ** 2
        ) + (1 / 3) * (
            (raman_tensor[l][0, 0] - raman_tensor[l][1, 1]) ** 2
            + (raman_tensor[l][0, 0] - raman_tensor[l][2, 2]) ** 2
            + (raman_tensor[l][1, 1] - raman_tensor[l][2, 2]) ** 2
        )
    return rotation_invariants


def get_raman_intensities(
    eigenenergies: np.ndarray, eigenmodes: np.ndarray, rotation_invariants: np.ndarray
) -> (np.ndarray, np.ndarray):
    def bose_einstein(omega, T=300):
        k_B = 0.695034800  # in cm^-1/K
        return 1 / (np.exp(omega / (k_B * T)) - 1)

    nmodes = eigenenergies.shape[0]
    omega_I = 18796.9987  # in cm^-1 for 532 nm laser
    Gamma = 10  # broadening in cm^-1
    c = 137.0359997566  # in A.U.
    omega_min, omega_max = 100, 1500
    omegas = np.linspace(omega_min, omega_max, 2000)
    C_array = np.zeros((nmodes, len(omegas)))

    for l in range(nmodes):
        omega_l = eigenenergies[l]
        C = (omega_l - omega_I) ** 4 / (2 * omega_l * c**4)
        C *= bose_einstein(omega_l) + 1
        C *= Gamma / ((omegas - omega_l) ** 2 + Gamma**2)
        C_array[l] = C

    I_array = np.zeros((nmodes, len(omegas)))
    for l in range(nmodes):
        I_array[l] = (
            2
            * np.pi
            * C_array[l]
            * (
                (10 * rotation_invariants[l, 0] + 4 * rotation_invariants[l, 2])
                + (5 * rotation_invariants[l, 1] + 3 * rotation_invariants[l, 2])
            )
        )
    return omegas, I_array.sum(0)


def write_raman_spectra(
    omegas: np.ndarray, raman_intensities: np.ndarray, outname: str
) -> None:
    with open(outname, "w") as f:
        f.write("# File written by the predict_spectrum.py script.\n")
        f.write("# frequency (cm^-1)\tIntensity\n")
        for e, i in zip(omegas, raman_intensities):
            f.write(f"{e:.5e}\t{i:.5e}\n")


def read_and_clean_phonons(
    dft_phonon_file: str,
) -> (np.ndarray, np.ndarray, np.ndarray):
    with h5py.File(args.dft_phonon_file) as f:
        eigenmodes = f["eigenmodes"][:]
        eigenenergies = f["eigenenergies"][:]

    non_translation_modes_idx = np.where(eigenenergies > 1)[0]
    return (
        eigenmodes[non_translation_modes_idx],
        eigenenergies[non_translation_modes_idx],
        non_translation_modes_idx,
    )


def main(args):
    atoms = read(args.pos_file)

    eigenmodes, eigenenergies, non_translation_modes_idx = read_and_clean_phonons(
        args.dft_phonon_file
    )

    ml_effective_charges = np.load(args.charge_path)
    dielectric = np.load(args.die_path)
    ml_raman_tensor = np.load(args.raman_tensor_path)[non_translation_modes_idx]
    if args.dft_raman_tensor_path is not None:
        dft_raman_tensor = np.load(args.dft_raman_tensor_path)[
            non_translation_modes_idx
        ]

    assert ml_effective_charges.shape[0] == len(atoms)

    squared_splittings = get_squared_loto_splittings(
        atoms, ml_effective_charges, dielectric
    )
    loto_splitting, loto_error, ml_eigenenergies = get_loto_splitting(
        squared_splittings, eigenmodes, eigenenergies
    )
    print(
        f"""
        Predicted LO-TO splitting is {np.mean(loto_splitting):3.2f} cm-1,
        with an error of {np.mean(np.abs(loto_error)):3.2f} cm-1.
        """
    )

    ml_rotation_invariants = get_rotation_invariants(ml_raman_tensor)
    ml_omegas, ml_raman_intensities = get_raman_intensities(
        ml_eigenenergies, eigenmodes, ml_rotation_invariants
    )
    write_raman_spectra(ml_omegas, ml_raman_intensities, "ml_raman_spec.dat")

    if args.dft_raman_tensor_path is not None:
        dft_rotation_invariants = get_rotation_invariants(dft_raman_tensor)
        dft_omegas, dft_raman_intensities = get_raman_intensities(
            eigenenergies, eigenmodes, dft_rotation_invariants
        )
        write_raman_spectra(dft_omegas, dft_raman_intensities, "dft_raman_spec.dat")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
