import torch
from collections import OrderedDict
from torch_scatter import scatter
from radnet.utils import pbc_round, DeterministicLinear
import numpy as np


class RadNet(torch.nn.Module):
    def __init__(
        self,
        cut_off=1.852 / 2,
        shape=(15, 15, 15),
        sigma=0.2,
        n_outputs=6,
        atom_types=None,
        input_normalization=True,
        cutoff_filter="erfc",
        biased_filters=False,
        bias_cell_lims=None,
        device="cpu",
    ):
        super(RadNet, self).__init__()
        self.cut_off = cut_off
        self.shape = shape
        self.sigma = sigma
        self.n_outputs = n_outputs
        self.atom_types = atom_types
        self.input_normalization = bool(input_normalization)

        # Set up real space grid
        self.x = torch.linspace(-cut_off, cut_off, shape[0])
        self.y = torch.linspace(-cut_off, cut_off, shape[1])
        self.z = torch.linspace(-cut_off, cut_off, shape[2])
        X, Y, Z = torch.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.X = X.to(device)
        self.Y = Y.to(device)
        self.Z = Z.to(device)

        # Biased filters to learn rotations of the cell
        self.biased_filters = bool(biased_filters)
        if self.biased_filters:
            self.bias_cell_lims = bias_cell_lims
            self.bias_min = 1.0
            self.bias_max = 4.0
            self.bias_slope = (self.bias_max - self.bias_min) / (
                self.bias_cell_lims[1] - self.bias_cell_lims[0]
            )

        self._setup_network()

        if atom_types is not None:
            test_images = torch.empty((len(atom_types),) + self.shape)
            for i, at in enumerate(atom_types):
                dr = self.X**2 + self.Y**2 + self.Z**2
                test_images[i] = at * torch.exp(-0.5 * dr / self.sigma**2)
            self.input_mean = torch.mean(test_images).to(device)
            self.input_std = torch.std(test_images).to(device)
            self.input_abs_max = torch.max(torch.abs(test_images)).to(device)

        assert cutoff_filter.lower() in [
            "erfc",
            "hard",
        ], f'You supplied a cutoff filter that is not implemented. Use "erfc" or "hard"'

        if cutoff_filter.lower() == "erfc":
            filters = (
                torch.erfc(
                    (self.X**2 + self.Y**2 + self.Z**2) ** 0.5 - self.cut_off
                )
                / 2.0
            )
        elif cutoff_filter.lower() == "hard":
            filters = torch.where(
                (self.X**2 + self.Y**2 + self.Z**2) ** 0.5 < self.cut_off,
                1.0,
                0.0,
            )
        if self.biased_filters:
            self.cutoff_filter = filters
        else:
            self.filter = filters

    def _setup_network(self):
        def create_layers(biased_filters):
            layersDict = OrderedDict()
            in_chan = 1
            for n in range(2):
                layersDict["conv_red_" + str(n)] = torch.nn.Conv3d(
                    in_chan, 64, 3, stride=1, padding=1
                )
                layersDict["conv_red_" + str(n) + "_elu"] = torch.nn.ELU()
                in_chan = 64

            for n in range(2, 6):
                layersDict["conv_red_" + str(n)] = torch.nn.Conv3d(
                    in_chan, 16, 3, stride=1, padding=1
                )
                layersDict["conv_red_" + str(n) + "_elu"] = torch.nn.ELU()
                in_chan = 16

            for n in range(6, 7):
                layersDict["conv_red_" + str(n)] = torch.nn.Conv3d(
                    in_chan, 64, 3, stride=2, padding=1
                )
                layersDict["conv_red_" + str(n) + "_elu"] = torch.nn.ELU()
                in_chan = 64

            for n in range(7, 11):
                layersDict["conv_red_" + str(n)] = torch.nn.Conv3d(
                    in_chan, 32, 3, stride=1, padding=1
                )
                layersDict["conv_red_" + str(n) + "_elu"] = torch.nn.ELU()
                in_chan = 32
            layersDict["flatten"] = torch.nn.Flatten()
            layersDict["fc1"] = DeterministicLinear(
                in_chan
                * (
                    self.shape[0] // 2 + 1
                    if self.shape[0] % 2 == 1
                    else self.shape[0] // 2
                )
                * (
                    self.shape[1] // 2 + 1
                    if self.shape[1] % 2 == 1
                    else self.shape[1] // 2
                )
                * (
                    self.shape[2] // 2 + 1
                    if self.shape[2] % 2 == 1
                    else self.shape[2] // 2
                ),
                1024,
            )
            layersDict["fc1_ELU"] = torch.nn.ELU()
            if biased_filters:
                layersDict["fc2"] = DeterministicLinear(1024, 1)
            else:
                layersDict["fc2"] = DeterministicLinear(1024, self.n_outputs)

            return layersDict

        if not self.biased_filters:
            layersDict = create_layers(biased_filters=False)
            self.model = torch.nn.Sequential(layersDict)
        else:
            if self.n_outputs == 3:
                layersDicts = [
                    create_layers(biased_filters=True),
                    create_layers(biased_filters=True),
                    create_layers(biased_filters=True),
                ]
                self.models = torch.nn.ModuleList(
                    [torch.nn.Sequential(lay) for lay in layersDicts]
                )
            else:
                raise NotImplementedError()

    def _get_distances(self, pos, neighbors, cell):
        # convert to scaled positions and handle PBCs
        scaled_pos = torch.linalg.solve(torch.transpose(cell, 1, 2), pos)
        scaled_neighbors = torch.transpose(
            torch.linalg.solve(
                torch.transpose(cell, 1, 2), torch.transpose(neighbors, 1, 2)
            ),
            1,
            2,
        )
        drs = scaled_pos.reshape(-1, 1, 3) - scaled_neighbors
        Rx = drs[..., 0].flatten()
        Ry = drs[..., 1].flatten()
        Rz = drs[..., 2].flatten()
        Rx -= pbc_round(Rx)
        Ry -= pbc_round(Ry)
        Rz -= pbc_round(Rz)
        Rs_scaled = torch.column_stack((Rx, Ry, Rz))
        Rs_scaled = Rs_scaled.reshape(drs.shape)
        Rs = Rs_scaled @ cell

        return Rs

    def _make_english_muffin(self, pos, Z, neighbors, use_neighbors):
        n_images = pos.shape[0]
        ems = torch.zeros((n_images,) + self.shape, device=pos.device)
        for i in range(n_images):
            p = pos[i]
            n = neighbors[i]
            n = n[use_neighbors[i].bool(), :]
            dr = p - n
            n_these_Rs = dr.shape[0]
            dx2 = (
                dr[:, 0].unsqueeze(-1) - torch.stack([self.X.flatten()] * n_these_Rs)
            ) ** 2
            dy2 = (
                dr[:, 1].unsqueeze(-1) - torch.stack([self.Y.flatten()] * n_these_Rs)
            ) ** 2
            dz2 = (
                dr[:, 2].unsqueeze(-1) - torch.stack([self.Z.flatten()] * n_these_Rs)
            ) ** 2
            ems[i] = (
                (Z[i] * torch.exp(-0.5 * (dx2 + dy2 + dz2) / (self.sigma**2)))
                .sum(0)
                .reshape(self.shape)
            )
        return ems

    def _get_bias_lims(self, pos):
        cut_min, cut_max = pos - self.cut_off, pos + self.cut_off
        lin_min = (cut_min - self.bias_cell_lims[0]) * self.bias_slope + self.bias_min
        lin_max = (cut_max - self.bias_cell_lims[0]) * self.bias_slope + self.bias_min
        return lin_min, lin_max

    def _apply_filters(self, ems, bias_lims=None):
        if self.biased_filters:
            device = ems.device
            bias_filters = [
                torch.zeros_like(ems),
                torch.zeros_like(ems),
                torch.zeros_like(ems),
            ]
            for i, em in enumerate(ems):
                biases = [
                    torch.linspace(
                        bias_lims[0][i][dim],
                        bias_lims[1][i][dim],
                        self.shape[dim],
                        device=device,
                    )
                    for dim in range(3)
                ]
                grid_biases = torch.meshgrid(
                    biases[0], biases[1], biases[2], indexing="ij"
                )
                for dim in range(3):
                    bias_filters[dim][i] = grid_biases[dim]

            return [self.cutoff_filter * bias * ems for bias in bias_filters]
        else:
            return self.filter * ems

    def forward(self, pos, Z, neighbors, use_neighbors, index):
        ems = self._make_english_muffin(pos, Z, neighbors, use_neighbors)
        if self.atom_types and self.input_normalization:
            ems -= self.input_mean
            ems /= self.input_std
            ems /= self.input_abs_max

        if self.biased_filters:
            bias_lims = self._get_bias_lims(pos)
            filtered_ems = self._apply_filters(ems, bias_lims=bias_lims)
        else:
            filtered_ems = self._apply_filters(ems)

        if self.biased_filters:
            outs = torch.empty(
                (torch.max(index) + 1, self.n_outputs), device=ems.device
            )
            for dim, (filtered_ems_dim, model_dim) in enumerate(
                zip(filtered_ems, self.models)
            ):
                inter_outs = model_dim(filtered_ems_dim.unsqueeze(1))
                outs[:, dim] = scatter(inter_outs, index, dim=0, reduce="add").squeeze()
        else:
            inter_outs = self.model(filtered_ems.unsqueeze(1))
            outs = scatter(inter_outs, index, dim=0, reduce="add")
        return outs
