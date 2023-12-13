import argparse
import numpy as np
import os
import subprocess


def create_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for RadNet prediction of properties for raman spectra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode",
        type=str,
        choices=["rcut", "shape"],
        help="Chooses which parameters varies between models.",
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


def read_value(prediction: str) -> np.array:
    if prediction == "pol":
        output = np.load("polarization.npy").reshape(3)
        os.remove("polarization.npy")
        return output

    elif prediction == "effch":
        output = np.load("effective_charges.npy").reshape(-1, 3, 3)
        os.remove("effective_charges.npy")
        return output
    else:
        raise NotImplementedError()


def main(args):
    workdir = os.getcwd()
    n_models = 5

    if args.mode == "rcut":
        rcuts = ["1.5", "2.0", "2.5", "3.0", "3.5", "4.0"]
        script_dir = os.path.dirname(__file__)
        base_command = (
            f"python {script_dir}/../../predict_raman.py {os.path.join(workdir, args.pos_file)} {args.prediction} "
            f"--n_outputs {args.n_outputs} --sigma {args.sigma} --device {args.device} "
            f"--image_shape {args.image_shape[0]} {args.image_shape[1]} {args.image_shape[2]} "
            f"--save_results "
        )

        all_values = []
        for rcut in rcuts:
            all_values.append([])
            for i in range(1, n_models + 1):
                model_dir = args.saved_model_dir + f"{rcut}/{i}/"
                os.chdir(model_dir)

                model_name = "best.torch"
                command = (
                    base_command + f"--rcut {rcut} --saved_model_path {model_name}"
                )
                out = subprocess.run(command.split(), capture_output=True, text=True)
                if out.returncode == 0:
                    value = read_value(args.prediction)
                    all_values[-1].append(value)
                else:
                    print("There was a problem with the subprocess!")
                    raise RuntimeError(out.stderr)

    elif args.mode == "shape":
        shapes = ["6", "9", "12", "15", "18"]
        script_dir = os.path.dirname(__file__)
        base_command = (
            f"python {script_dir}/../../predict_raman.py {os.path.join(workdir, args.pos_file)} {args.prediction} "
            f"--n_outputs {args.n_outputs} --sigma {args.sigma} --device {args.device} "
            f"--rcut {args.rcut} --save_results "
        )

        all_values = []
        for shape in shapes:
            all_values.append([])
            for i in range(1, n_models + 1):
                model_dir = args.saved_model_dir + f"{shape}/{i}/"
                os.chdir(model_dir)

                model_name = "best.torch"
                command = (
                    base_command
                    + f"--image_shape {shape} {shape} {shape} --saved_model_path {model_name}"
                )
                out = subprocess.run(command.split(), capture_output=True, text=True)
                if out.returncode == 0:
                    value = read_value(args.prediction)
                    all_values[-1].append(value)
                else:
                    print("There was a problem with the subprocess!")
                    raise RuntimeError(out.stderr)
    else:
        raise RuntimeError("Run mode {args.mode} is not recognized.")

    os.chdir(workdir)
    all_values = np.array(all_values)
    np.save(args.save_name, all_values)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
