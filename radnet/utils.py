import numpy as np
import torch
import torch.nn


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


def generate_random_z_axis_rotation():
    R = np.eye(3)
    x1 = np.random.rand()
    R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
    R[0, 1] = -np.sin(2 * np.pi * x1)
    R[1, 0] = np.sin(2 * np.pi * x1)
    return R


def generate_random_3D_rotation_matrix():
    r"Algorithm from James Avro, https://doi.org/10.1016/B978-0-08-050755-2.50034-8"

    x2 = 2 * np.pi * np.random.rand()
    x3 = np.random.rand()

    R = generate_random_z_axis_rotation()
    v = np.array([np.cos(x2) * np.sqrt(x3), np.sin(x2) * np.sqrt(x3), np.sqrt(1 - x3)])
    H = np.eye(3) - (2 * np.outer(v, v))
    M = -(H @ R)
    return M


def pbc_round(tensor):
    i = tensor.int()
    bools = abs(tensor - i) >= 0.5
    vals = torch.where(torch.logical_and(bools, tensor > 0), i + 1, i)
    vals = torch.where(torch.logical_and(bools, tensor < 0), i - 1, i)
    return vals


# Not sure why this is needed, but without it the Linear layers at the
# end of the model returns different values for equivalent atoms when batch_size > 1
# for some GPU configurations.
# This seems to solve it for now.
class DeterministicLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        y = torch.matmul(x, self.weight.T.unsqueeze(0))[0] + self.bias
        return y


def get_symmetries_array(n_symmetries):
    if n_symmetries == 24:
        symmetries_array = np.zeros((24, 3, 3))
        symmetries_array[0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        symmetries_array[1] = np.array([[0, 0, 1], [-1, -1, -1], [1, 0, 0]])
        symmetries_array[2] = np.array([[-1, -1, -1], [0, 0, 1], [0, 1, 0]])
        symmetries_array[3] = np.array([[0, 1, 0], [1, 0, 0], [-1, -1, -1]])
        symmetries_array[4] = np.array([[-1, -1, -1], [0, 1, 0], [0, 0, 1]])
        symmetries_array[5] = np.array([[0, 1, 0], [-1, -1, -1], [1, 0, 0]])
        symmetries_array[6] = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        symmetries_array[7] = np.array([[0, 0, 1], [1, 0, 0], [-1, -1, -1]])
        symmetries_array[8] = np.array([[-1, -1, -1], [0, 1, 0], [1, 0, 0]])
        symmetries_array[9] = np.array([[0, 1, 0], [-1, -1, -1], [0, 0, 1]])
        symmetries_array[10] = np.array([[1, 0, 0], [0, 0, 1], [-1, -1, -1]])
        symmetries_array[11] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        symmetries_array[12] = np.array([[1, 0, 0], [0, 1, 0], [-1, -1, -1]])
        symmetries_array[13] = np.array([[0, 0, 1], [-1, -1, -1], [0, 1, 0]])
        symmetries_array[14] = np.array([[-1, -1, -1], [0, 0, 1], [1, 0, 0]])
        symmetries_array[15] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        symmetries_array[16] = np.array([[0, 0, 1], [0, 1, 0], [-1, -1, -1]])
        symmetries_array[17] = np.array([[1, 0, 0], [-1, -1, -1], [0, 1, 0]])
        symmetries_array[18] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        symmetries_array[19] = np.array([[-1, -1, -1], [1, 0, 0], [0, 0, 1]])
        symmetries_array[20] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        symmetries_array[21] = np.array([[1, 0, 0], [-1, -1, -1], [0, 0, 1]])
        symmetries_array[22] = np.array([[0, 1, 0], [0, 0, 1], [-1, -1, -1]])
        symmetries_array[23] = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0]])
    elif n_symmetries == 6:
        symmetries_array = np.zeros((6, 3, 3))
        symmetries_array[0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        symmetries_array[1] = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        symmetries_array[2] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        symmetries_array[3] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        symmetries_array[4] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        symmetries_array[5] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    elif n_symmetries == 2:
        symmetries_array = np.zeros((2, 3, 3))
        symmetries_array[0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        symmetries_array[1] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    else:
        raise NotImplementedError()
    return symmetries_array
