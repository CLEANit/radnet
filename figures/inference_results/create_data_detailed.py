import argparse
import numpy as np
import os
import pickle
import subprocess


def create_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for RadNet inference on test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datadir",
        type=str,
        help="Path to the folder containing the datasets.",
    )
    parser.add_argument(
        "--sigma", type=float, default=0.5, help="sigma value used for the gaussians"
    )
    parser.add_argument(
        "--saved_model_dir",
        type=str,
        help="Path to the directory containing the saved models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device.",
    )
    parser.add_argument(
        "--save_name", type=str, help="Name of the file for saved values."
    )
    return parser


def read_values() -> np.array:
    with open("detailed_results.pkl", "rb") as f:
        data = pickle.load(f)
    trues = data["trues"]
    preds = data["preds"]
    return trues, preds


def main(args):
    workdir = os.getcwd()

    script_dir = os.path.dirname(__file__)
    all_results = {}

    os.chdir(args.saved_model_dir)

    for material in ["BN", "GaAs"]:
        rcut = 4.5 if material == "BN" else 7.0
        image_shape = (12, 12, 12) if material == "BN" else (15, 15, 15)
        for prop in ["polarization", "dielectric"]:
            n_outputs = 3 if prop == "polarization" else 6

            dataset_path = os.path.join(args.datadir, f"{material}_test_100_{prop}.h5")
            base_command = f"python {script_dir}/../../inference.py {dataset_path} "
            base_options = (
                f" --n_outputs {n_outputs} --sigma {args.sigma} --device {args.device} --rcut {rcut} "
                f"--image_shape {image_shape[0]} {image_shape[1]} {image_shape[2]} --detailed "
            )

            os.chdir(f"{material}/{prop}/")
            model_name = "best.torch"
            command = base_command + base_options + f"--saved_model_path {model_name}"
            out = subprocess.run(command.split(), capture_output=True, text=True)
            if out.returncode == 0:
                trues, preds = read_values()
                all_results[f"{material}_{prop}_trues"] = trues
                all_results[f"{material}_{prop}_preds"] = preds
            else:
                print("There was a problem with the subprocess!")
                raise RuntimeError(out.stderr)
            os.chdir("../../")

    os.chdir(workdir)
    save_name = (
        args.save_name if args.save_name.endswith(".pkl") else args.save_name + ".pkl"
    )
    with open(save_name, "wb") as f:
        pickle.dump(all_results, f)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
