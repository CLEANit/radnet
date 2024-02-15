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
        "--material",
        type=str,
        choices=("BN", "GaAs"),
        help="Either BN or GaAs.",
    )
    parser.add_argument(
        "--property",
        type=str,
        choices=("polarization", "dielectric"),
        help="Either polarization or dielectric.",
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
        "--rcut", type=float, default=4.0, help="Cut off radius (in Angstrom)"
    )
    parser.add_argument(
        "--sigma", type=float, default=0.5, help="sigma value used for the gaussians"
    )
    parser.add_argument(
        "--saved_model_dir",
        type=str,
        help="Path to the directory containing the saved models.",
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
    with open("inference_results.txt", "r") as f:
        data = f.read()
    data = [float(d) for d in data.split()]
    os.remove("inference_results.txt")
    return data


def main(args):
    workdir = os.getcwd()

    n_models = 5
    sizes = (
        ["100", "250", "500", "900"] if args.material == "BN" else ["100", "250", "500"]
    )
    script_dir = os.path.dirname(__file__)
    base_command = f"python {script_dir}/../../inference.py "
    base_options = (
        f" --n_outputs {args.n_outputs} --sigma {args.sigma} --device {args.device} --rcut {args.rcut} "
        f"--image_shape {args.image_shape[0]} {args.image_shape[1]} {args.image_shape[2]} "
    )

    all_maes, all_rmses = [], []
    for size in sizes:
        print(size)
        all_maes.append([])
        all_rmses.append([])
        dataset_path = os.path.join(
            args.datadir, f"{args.material}_test_100_{args.property}.h5"
        )
        for i in range(1, n_models + 1):
            print(i)
            model_dir = args.saved_model_dir + f"{size}/{i}/"
            os.chdir(model_dir)

            model_name = "best.torch"
            command = (
                base_command
                + dataset_path
                + base_options
                + f"--saved_model_path {model_name}"
            )
            out = subprocess.run(command.split(), capture_output=True, text=True)
            if out.returncode == 0:
                mae, rmse = read_values()
                all_maes[-1].append(mae)
                all_rmses[-1].append(rmse)
            else:
                print("There was a problem with the subprocess!")
                raise RuntimeError(out.stderr)

    os.chdir(workdir)
    all_maes = np.array(all_maes)
    all_rmses = np.array(all_rmses)

    save_name = (
        args.save_name if args.save_name.endswith(".pkl") else args.save_name + ".pkl"
    )
    with open(save_name, "wb") as f:
        pickle.dump({"maes": all_maes, "rmses": all_rmses}, f)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
