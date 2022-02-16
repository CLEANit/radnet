import torch
from collections import OrderedDict
from torch_scatter import scatter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pbc_round(input):
     i = input.int()
     bools = abs(input - i) >= 0.5
     vals = torch.where(torch.logical_and(bools, input > 0), i + 1, i)
     vals = torch.where(torch.logical_and(bools, input < 0), i - 1, i)
     return vals

class RadNet(torch.nn.Module):
    def __init__(self, cut_off=1.852 / 2, shape=(27, 27, 27), sigma=0.2, n_outputs=6, atom_types=None, cutoff_filter='erfc'):
        super(RadNet, self).__init__()
        self.cut_off = cut_off
        self.shape = shape
        self.sigma = sigma
        self.n_outputs = n_outputs
        self.atom_types = atom_types
        self.x = torch.linspace(-cut_off, cut_off, shape[0])
        self.y = torch.linspace(-cut_off, cut_off, shape[1])
        self.z = torch.linspace(-cut_off, cut_off, shape[2])
        X, Y, Z = torch.meshgrid(self.x, self.y, self.z)
        self.X = X.to(device)
        self.Y = Y.to(device)
        self.Z = Z.to(device)
        self._setup_network()

        if atom_types is not None:
            test_images = torch.empty((len(atom_types),) + self.shape)
            for i, at in enumerate(atom_types):
                dr = self.X**2 + self.Y**2 + self.Z**2
                test_images[i] = at * torch.exp(-0.5 * dr / self.sigma**2)
            self.input_mean = torch.mean(test_images).to(device)
            self.input_std = torch.std(test_images).to(device)
            self.input_abs_max = torch.max(torch.abs(test_images)).to(device)

        assert cutoff_filter.lower() in ['erfc', 'hard'], f'You supplied a cutoff filter that is not implemented. Use "erfc" or "hard"'

        if cutoff_filter.lower() == 'erfc':
            self.filter = torch.erfc((self.X**2 + self.Y**2 + self.Z**2)**0.5 - self.cut_off) / 2. 
        elif cutoff_filter.lower() == 'hard':
            self.filter = torch.where((self.X**2 + self.Y**2 + self.Z**2)**0.5 < self.cut_off, 1.0, 0.0)


    def _setup_network(self):
        layers = OrderedDict()
        in_chan = 1
        for n in range(2):
            layers['conv_red_' + str(n)] = torch.nn.Conv3d(in_chan, 64, 3, stride=1, padding=1)
            layers['conv_red_' + str(n) + '_elu'] = torch.nn.ELU()
            in_chan = 64
        
        for n in range(2, 6):
            layers['conv_red_' + str(n)] = torch.nn.Conv3d(in_chan, 16, 3, stride=1, padding=1)
            layers['conv_red_' + str(n) + '_elu'] = torch.nn.ELU()
            in_chan = 16

        for n in range(6,7):
            layers['conv_red_' + str(n)] = torch.nn.Conv3d(in_chan, 64, 3, stride=2, padding=1)
            layers['conv_red_' + str(n) + '_elu'] = torch.nn.ELU()
            in_chan = 64

        for n in range(7,11):
            layers['conv_red_' + str(n)] = torch.nn.Conv3d(in_chan, 32, 3, stride=1, padding=1)
            layers['conv_red_' + str(n) + '_elu'] = torch.nn.ELU()
            in_chan = 32
        layers['flatten'] = torch.nn.Flatten()    
        layers['fc1'] = torch.nn.Linear(in_chan * (self.shape[0] // 2 + 1) * (self.shape[1] // 2 + 1) * (self.shape[2] // 2 + 1), 1024 )
        layers['fc1_ELU'] = torch.nn.ELU()
        layers['fc2'] = torch.nn.Linear(1024, self.n_outputs)
        self.model = torch.nn.Sequential(layers)


    def _get_distances(self, pos, neighbors, cell):

        # convert to scaled positions and handle PBCs
        scaled_pos = torch.linalg.solve(torch.transpose(cell, 1, 2), pos)
        scaled_neighbors = torch.transpose(torch.linalg.solve(torch.transpose(cell, 1, 2), torch.transpose(neighbors, 1, 2)), 1, 2)
        print(scaled_neighbors)
        drs = scaled_pos.reshape(-1, 1, 3) - scaled_neighbors
        Rx = drs[..., 0].flatten()
        Ry = drs[..., 1].flatten()
        Rz = drs[..., 2].flatten()
        Rx -= pbc_round(Rx)
        Ry -= pbc_round(Ry)
        Rz -= pbc_round(Rz)
        Rs_scaled = torch.column_stack((Rx, Ry, Rz))
        Rs_scaled = Rs_scaled.reshape(drs.shape)
        print(Rs_scaled)
        Rs = Rs_scaled @ cell

        return Rs

    def _make_english_muffin(self, pos, Z, neighbors, use_neighbors, cell, index):
        n_images = pos.shape[0]
        ems = torch.zeros((n_images,) + self.shape, device=pos.device)
        for i in range(n_images):
            p = pos[i]
            n = neighbors[i]
            n = n[use_neighbors[i].bool(), :]
            dr = p - n
            n_these_Rs = dr.shape[0]
            dx2 = (dr[:, 0].unsqueeze(-1) - torch.stack([self.X.flatten()]*n_these_Rs))**2
            dy2 = (dr[:, 1].unsqueeze(-1) - torch.stack([self.Y.flatten()]*n_these_Rs))**2
            dz2 = (dr[:, 2].unsqueeze(-1) - torch.stack([self.Z.flatten()]*n_these_Rs))**2
            ems[i] = (Z[i] * torch.exp(-0.5 * (dx2 + dy2 + dz2) / (self.sigma**2))).sum(0).reshape(self.shape)
        return ems
        
    def _apply_filter(self, ims):
        return self.filter * ims

    def forward(self, pos, Z, neighbors, use_neighbors, cell, index):

        ems = self._make_english_muffin(pos, Z, neighbors, use_neighbors, cell, index)
        if self.atom_types:
            ems -= self.input_mean
            ems /= self.input_std
            ems /= self.input_abs_max
        filtered_ems = self._apply_filter(ems)
        import matplotlib.pyplot as plt, h5py
        # f = h5py.File('images_die_3.5.h5')
        for i, em in enumerate(filtered_ems):
            fig, ax = plt.subplots(1, 3)
            # im = ax[0].imshow(f['sliced_ems'][index[i],i % 2].sum(-1))
            # plt.colorbar(im, ax=ax[0])
            im = ax[1].imshow(em.sum(-1).detach().numpy())
            plt.colorbar(im, ax=ax[1])
            # im =  ax[2].imshow(f['sliced_ems'][index[i], i % 2].sum(-1) - em.sum(-1).detach().numpy())
            plt.colorbar(im, ax=ax[2])
            plt.show()
        exit()
        inter_outs = self.model(filtered_ems.unsqueeze(1))
        outs = scatter(inter_outs, index, dim=0, reduce='add')
        return outs
