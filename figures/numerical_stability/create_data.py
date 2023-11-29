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


def extract_value(prediction: str, output: str) -> np.array:
    if prediction == "pol":
        output = output.split("\n")

        for i, line in enumerate(output):
            if line.startswith("Polarization"):
                value_index = i + 1
        output = output[value_index].split()[:-1]
        output[0] = output[0].lstrip("tensor([[").rstrip(",")
        output[1] = output[1].rstrip(",")
        output[2] = output[2].rstrip("]],")
        output = np.array([float(o) for o in output])
        return output
    else:
        raise NotImplementedError()


def main(args):
    n_models = 2
    rcuts = ["2.0", "4.0"]
    script_dir = os.path.dirname(__file__)
    base_command = (
        f"python {script_dir}/../../predict_raman.py {args.pos_file} {args.prediction} "
        f"--n_outputs {args.n_outputs} --sigma {args.sigma} --device {args.device} "
        f"--image_shape {args.image_shape[0]} {args.image_shape[1]} {args.image_shape[2]} "
    )

    all_values = []
    for rcut in rcuts:
        all_values.append([])
        for i in range(1, n_models + 1):
            model_path = args.saved_model_dir + f"{rcut}/best_{i}.torch"
            command = base_command + f"--rcut {rcut} --saved_model_path {model_path}"
            out = subprocess.run(command.split(), capture_output=True, text=True)
            if out.returncode == 0:
                value = extract_value(args.prediction, out.stdout)
                all_values[-1].append(value)
            else:
                print("There was a problem with the subprocess!")
                raise RuntimeError(out.stderr)

    all_values = np.array(all_values)
    np.save(args.save_name, all_values)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
