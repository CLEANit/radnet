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
        "datapath",
        type=str,
        help="Path to the dataset to evaluate.",
    )
    parser.add_argument(
        "--n_outputs", type=int, default=3, help="number of outputs in neural network"
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
    shapes = ["6", "9", "12", "15", "18"]
    script_dir = os.path.dirname(__file__)
    base_command = (
        f"python {script_dir}/../../inference.py {args.datapath} --n_outputs {args.n_outputs} "
        f"--sigma {args.sigma} --device {args.device} --rcut {args.rcut} "
    )

    all_maes, all_rmses = [], []
    for shape in shapes:
        print(shape)
        all_maes.append([])
        all_rmses.append([])
        for i in range(1, n_models + 1):
            print(i)
            model_dir = args.saved_model_dir + f"{shape}/{i}/"
            os.chdir(model_dir)

            model_name = "best.torch"
            command = (
                base_command
                + f"--image_shape {shape} {shape} {shape} --saved_model_path {model_name}"
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
