#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import pickle
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter


class QuantumDeepField(nn.Module):
    def __init__(self, device, N_orbitals,
                 dim, layer_functional, operation, N_output,
                 hidden_HK, layer_HK, use_d=False, density_only=False, exch_scale=1.0):
        super(QuantumDeepField, self).__init__()

        """All learning parameters of the QDF model."""
        self.coefficient = nn.Embedding(N_orbitals, dim)
        self.zeta = nn.Embedding(N_orbitals, 1)  # Orbital exponent.
        nn.init.ones_(self.zeta.weight)  # Initialize each zeta with one.
        
        self.W_pre_func = nn.Linear(1, dim)
        self.W_functional = nn.ModuleList([nn.Linear(dim, dim)
                                           for _ in range(layer_functional)])
        self.W_property = nn.Linear(dim, N_output)
        self.W_density = nn.Linear(1, hidden_HK)
        self.W_HK = nn.ModuleList([nn.Linear(hidden_HK, hidden_HK)
                                   for _ in range(layer_HK)])
        self.W_potential = nn.Linear(hidden_HK, 1)

        self.device = device
        self.dim = dim
        self.layer_functional = layer_functional
        self.operation = operation
        self.layer_HK = layer_HK

        self.use_d = use_d
        self.density_only = density_only
        self.exch_scale = exch_scale

    def list_to_batch(self, xs, dtype=torch.FloatTensor, cat=None, axis=None):
        """Transform the list of numpy data into the batch of tensor data."""
        xs = [dtype(x).to(self.device) for x in xs]
        if cat:
            return torch.cat(xs, axis)
        else:
            return xs  # w/o cat (i.e., the list (not batch) of tensor data).

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0 and large value) for batch processing.
        For example, given a list of matrices [A, B, C],
        this function returns a new matrix [A00, 0B0, 00C],
        where 0 is the zero matrix (i.e., a block diagonal matrix).
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        pad_matrices = torch.full((M, N), pad_value, device=self.device)
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            matrix = torch.FloatTensor(matrix).to(self.device)
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def basis_matrix(self, atomic_orbitals,
                     distance_matrices, quantum_numbers, laplace=False):
        """Transform the distance matrix into a basis matrix,
        in which each element is d^(q-1)*e^(-z*d^2), where d is the distance,
        q is the principle quantum number, and z is the orbital exponent.
        We note that this is a simplified Gaussian-type orbital (GTO)
        in terms of the spherical harmonics.
        We simply normalize the GTO basis matrix using F.normalize in PyTorch.
        """
        zetas = torch.squeeze(self.zeta(atomic_orbitals))
        GTOs = (distance_matrices**(quantum_numbers-1) *
                torch.exp(-zetas**2*distance_matrices**2))
        
        # GTOs = F.normalize(GTOs, 2, 0)
        # # print(torch.sum(torch.t(GTOs)[0]**2))  # Normalization check.
        # return GTOs

        denom = GTOs.norm(2, 0, True).clamp_min(1e-12).expand_as(GTOs)
        n_GTOs = GTOs/denom
       
        if laplace == False:
            return n_GTOs
        else:
            """A laplace Gaussian-type orbital (GTO)        """
            l_GTOs = (distance_matrices**(quantum_numbers-3) *
                torch.exp(-zetas**2*distance_matrices**2)) * (4*zetas**4*distance_matrices**4
                -2*zetas**2*(2*quantum_numbers+1)*distance_matrices**2
                       +quantum_numbers*(quantum_numbers-1))
            l_GTOs=l_GTOs/denom
            return l_GTOs

    def LCAO(self, inputs, laplace=False):
        """Linear combination of atomic orbitals (LCAO)."""

        """Inputs."""
        (atomic_orbitals, distance_matrices,
         quantum_numbers, N_electrons, N_fields) = inputs

        """Cat or pad each input data for batch processing."""
        atomic_orbitals = self.list_to_batch(atomic_orbitals, torch.LongTensor)
        distance_matrices = self.pad(distance_matrices, 1e6)
        quantum_numbers = self.list_to_batch(quantum_numbers, cat=True, axis=1)
        N_electrons = self.list_to_batch(N_electrons)

        """Normalize the coefficients in LCAO."""
        coefficients = []
        for AOs in atomic_orbitals:
            coefs = F.normalize(self.coefficient(AOs), 2, 0)
            # print(torch.sum(torch.t(coefs)[0]**2))  # Normalization check.
            coefficients.append(coefs)
        coefficients = torch.cat(coefficients)
        atomic_orbitals = torch.cat(atomic_orbitals)

        """LCAO."""
        basis_matrix = self.basis_matrix(atomic_orbitals,
                                         distance_matrices, quantum_numbers)
        molecular_orbitals = torch.matmul(basis_matrix, coefficients)

        """We simply normalize the molecular orbitals
        and keep the total electrons of the molecule
        in learning the molecular orbitals.
        """
        split_MOs = torch.split(molecular_orbitals, N_fields)
        normalized_MOs = []

        denom_list = []
        for N_elec, MOs in zip(N_electrons, split_MOs):
            # MOs = torch.sqrt(N_elec/self.dim) * F.normalize(MOs, 2, 0)
            # # print(torch.sum(MOs**2), N_elec)  # Total electrons check.
            # normalized_MOs.append(MOs)
            denom = MOs.norm(2, 0, True).clamp_min(1e-12).expand_as(MOs)
            n_MTOs = MOs / denom
            MOs = torch.sqrt(N_elec / self.dim) * n_MTOs
            denom_list.append(denom)
            normalized_MOs.append(MOs)            
        if laplace == False:
            return torch.cat(normalized_MOs)
        else:
            l_basis_matrix= self.basis_matrix(atomic_orbitals,
                                         distance_matrices, quantum_numbers,laplace=laplace)
            l_molecular_orbitals = torch.matmul(l_basis_matrix, coefficients)
            l_split_MOs = torch.split(l_molecular_orbitals, N_fields)
            l_normalized_MOs = [] 
            index = 0
            for N_elec, MOs in zip(N_electrons, l_split_MOs):
                n_MTOs = MOs / denom_list[index]
                MOs = torch.sqrt(N_elec / self.dim) * n_MTOs
                l_normalized_MOs.append(MOs)
                index += 1
            return torch.cat(normalized_MOs), torch.cat(l_normalized_MOs)

    def functional(self, vectors, layers, operation, axis, use_d=False, density_only=False):
        """DNN-based energy functional."""
        if use_d or density_only:
            vectors = torch.relu(self.W_pre_func(vectors))
        for l in range(layers):
            vectors = torch.relu(self.W_functional[l](vectors))
        if operation == 'sum':
            vectors = [torch.sum(vs, 0) for vs in torch.split(vectors, axis)]
        if operation == 'mean':
            vectors = [torch.mean(vs, 0) for vs in torch.split(vectors, axis)]
        return torch.stack(vectors)

    def HKmap(self, scalars, layers):
        """DNN-based Hohenberg--Kohn map."""
        vectors = self.W_density(scalars)
        for l in range(layers):
            vectors = torch.relu(self.W_HK[l](vectors))
        return self.W_potential(vectors)

    def forward(self, data, train=False, target=None, predict=False):
        """Forward computation of the QDF model
        using the above defined functions.
        """

        idx, inputs, N_fields = data[0], data[1:6], data[5]

        if predict:  # For demo.
            with torch.no_grad():
                molecular_orbitals = self.LCAO(inputs)
                if self.use_d or self.density_only:
                    densities = torch.sum(molecular_orbitals ** 2, 1)
                    densities = torch.unsqueeze(densities, 1)
                    final_layer = self.functional(densities,
                                              self.layer_functional,
                                              self.operation, N_fields, use_d=self.use_d,density_only=self.density_only)
    
                    if self.density_only:
                        E_ = self.W_property(final_layer)
                    else:
                        V_n = self.list_to_batch(data[7], cat=True, axis=0)  # Correct V.
                        E_xcH = self.W_property(final_layer)
                        # E_ = E_xcH
                        d_n = 0
                        batch_num = E_xcH.size()[0]
                        E_n=torch.zeros_like(E_xcH)
                        E_k=torch.zeros_like(E_xcH)            
                        for i in range(batch_num):
                            d_ni=data[2][i].shape[0]
                            E_n[i] = torch.sum(V_n[d_n:d_n+d_ni] * densities[d_n:d_n+d_ni]) 
                            E_k[i]=  torch.sum(molecular_orbitals[d_n:d_n +d_ni] * l_molecular_orbitals[d_n:d_n + d_ni]  )/2
                            d_n += d_ni
                        E_ = self.exch_scale * E_xcH + E_n - E_k
                        # E_n = torch.sum(V_n * densities)
                        # E_ = E_xcH + E_n*self.en_scale
                else:
                    final_layer = self.functional(molecular_orbitals,
                                              self.layer_functional,
                                              self.operation, N_fields)
                    E_ = self.W_property(final_layer)
                return idx, E_

        elif train:
            if target == 'E':  # Supervised learning for energy.
                molecular_orbitals = self.LCAO(inputs)
                E = self.list_to_batch(data[6], cat=True, axis=0)  # Correct E.
                if self.density_only:
                    densities = torch.sum(molecular_orbitals ** 2, 1)
                    densities = torch.unsqueeze(densities, 1)
                    final_layer = self.functional(densities,
                                              self.layer_functional,
                                              self.operation, N_fields, density_only=True)
                else:
                    final_layer = self.functional(molecular_orbitals,
                                              self.layer_functional,
                                              self.operation, N_fields)
                E_ = self.W_property(final_layer)  # Predicted E.
                loss = F.mse_loss(E, E_)
                return loss
            if target == 'V':  # Unsupervised learning for potential.
                molecular_orbitals = self.LCAO(inputs)
                V = self.list_to_batch(data[7], cat=True, axis=0)  # Correct V.
                densities = torch.sum(molecular_orbitals**2, 1)
                densities = torch.unsqueeze(densities, 1)
                V_ = self.HKmap(densities, self.layer_HK)  # Predicted V.
                loss = F.mse_loss(V, V_)
                return loss
            if target == 'D':
                E = self.list_to_batch(data[6], cat=True, axis=0)  # Correct E.
                molecular_orbitals, l_molecular_orbitals = self.LCAO(inputs,laplace = True)
                V_n = self.list_to_batch(data[7], cat=True, axis=0)  # Correct V.
                densities = torch.sum(molecular_orbitals ** 2, 1)
                densities = torch.unsqueeze(densities, 1)
                final_layer = self.functional(densities,
                                            self.layer_functional,
                                            self.operation, N_fields, use_d=True)
                E_xcH = self.W_property(final_layer)

                d_n = 0
                batch_num = E_xcH.size()[0]
                E_n=torch.zeros_like(E_xcH)
                E_k=torch.zeros_like(E_xcH)            
                for i in range(batch_num):
                    d_ni=data[2][i].shape[0]
                    E_n[i] = torch.sum(V_n[d_n:d_n+d_ni] * densities[d_n:d_n+d_ni]) 
                    E_k[i]=  torch.sum(molecular_orbitals[d_n:d_n +d_ni] * l_molecular_orbitals[d_n:d_n + d_ni]  )/2
                    d_n += d_ni
                E_ = self.exch_scale * E_xcH + E_n - E_k 
                loss1 = F.mse_loss(E, E_)

                # #2 for each molecular's loss, we have an epsilon.we sum them to the total loss and backward         
                # E_n = torch.sum(V_n * densities)
                # # import pdb; pdb.set_trace()
                # E_ = E_xcH + E_n*self.en_scale
                # # E_ = E_xcH
                # loss1 = F.mse_loss(E, E_)
                        
                grad_v = []
                batch_num = len(data[2])            
                for i in range(batch_num):
                    grad_v.append(torch.autograd.grad(outputs=E_xcH[i], inputs=densities, retain_graph=True)[0])
                V_xcH = grad_v[0]
                dim_num = 0
                for i in range(batch_num):
                    V_xcH[dim_num:dim_num + data[2][i].shape[0]] = grad_v[i][dim_num:dim_num + data[2][i].shape[0]]
                    dim_num += data[2][i].shape[0]
                V = V_xcH +V_n
                
                loss2=0        
                mat = molecular_orbitals*V-1/2*l_molecular_orbitals
                dim=0
                for j in range(batch_num):
                    dim_mole=data[2][j].shape[0]
                    loss2 += torch.sum(torch.sum(torch.square(mat[dim:dim+dim_mole,:]),0)-torch.sum(torch.square(molecular_orbitals[dim:dim+dim_mole,:]*mat[dim:dim+dim_mole,:]),0)/torch.sum(torch.square(molecular_orbitals[dim:dim+dim_mole,:]),0)    )          
                    dim += dim_mole            
                return loss1, loss2/(batch_num)

        else:  # Test.
            with torch.no_grad():
                E = self.list_to_batch(data[6], cat=True, axis=0)
                molecular_orbitals, l_molecular_orbitals = self.LCAO(inputs,laplace = True)
                if self.use_d or self.density_only:
                    densities = torch.sum(molecular_orbitals ** 2, 1)
                    densities = torch.unsqueeze(densities, 1)
                    final_layer = self.functional(densities,
                                              self.layer_functional,
                                              self.operation, N_fields, use_d=self.use_d, density_only=self.density_only)
    
                    if self.density_only:
                        E_ = self.W_property(final_layer)
                    else:
                        V_n = self.list_to_batch(data[7], cat=True, axis=0)  # Correct V.
                        E_xcH = self.W_property(final_layer)
                        # E_ = E_xcH
                        d_n = 0
                        batch_num = E_xcH.size()[0]
                        E_n=torch.zeros_like(E_xcH)
                        E_k=torch.zeros_like(E_xcH)            
                        for i in range(batch_num):
                            d_ni=data[2][i].shape[0]
                            E_n[i] = torch.sum(V_n[d_n:d_n+d_ni] * densities[d_n:d_n+d_ni]) 
                            E_k[i]=  torch.sum(molecular_orbitals[d_n:d_n +d_ni] * l_molecular_orbitals[d_n:d_n + d_ni]  )/2
                            d_n += d_ni
                        E_ = self.exch_scale * E_xcH + E_n - E_k 

                        # E_n = torch.sum(V_n * densities)
                        # E_ = E_xcH + E_n*self.en_scale
                else:
                    final_layer = self.functional(molecular_orbitals,
                                              self.layer_functional,
                                              self.operation, N_fields)
                    E_ = self.W_property(final_layer)
                return idx, E, E_


class Trainer(object):
    def __init__(self, model, lr, lr_decay, step_size, lambdaV=1.0, use_d=False, lambdaD=0.01, orbit_lrdecay=1, load_path=""):
        self.model = model

        # resume from step1:
        def map_func(storage, location):
            return storage.cuda()
        if load_path:
            checkpoint = torch.load(load_path, map_location=map_func)
            self.model.load_state_dict(checkpoint)

        # split the parameter into two parts
        orbit_params = list(map(id, model.coefficient.parameters()))
        orbit_params += list(map(id, model.zeta.parameters()))
        
        rest_params = filter(lambda x: id(x) not in orbit_params, model.parameters())


        if orbit_lrdecay == 0:
        # freeze the orbit parameter
            for name, p in model.named_parameters():
                if name.startswith("coefficient") or name.startswith("zeta"):
                    p.requires_grad = False
                    print("Freeze the weight of {}".format(name))
        
            self.optimizer = optim.Adam(rest_params, lr)
        else:
            self.optimizer = optim.Adam([{'params':rest_params,'lr':lr},
                        {'params':model.coefficient.parameters(),'lr':lr * orbit_lrdecay},
                        {'params':model.zeta.parameters(),'lr':lr * orbit_lrdecay}
                        ])

        # self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size, lr_decay)
        self.lambdaV = lambdaV

        self.use_d = use_d
        self.lambdaD = lambdaD

    def optimize(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, dataloader):
        """Minimize two loss functions in terms of E and V."""
        losses_E, losses_V = 0, 0
        for data in dataloader:
            if self.use_d:
                loss_E, loss_V = self.model.forward(data, train=True, target='D')
                total_loss = loss_E + self.lambdaD * loss_V
                self.optimize(total_loss, self.optimizer)
                losses_E += loss_E.item()
                losses_V += loss_V.item()
            else:
                loss_E = self.model.forward(data, train=True, target='E')
                self.optimize(loss_E, self.optimizer)
                losses_E += loss_E.item()
                loss_V = self.model.forward(data, train=True, target='V')
                loss_V *= self.lambdaV
                self.optimize(loss_V, self.optimizer)
                losses_V += loss_V.item()
        self.scheduler.step()
        return losses_E, losses_V


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataloader, time=False):
        N = sum([len(data[0]) for data in dataloader])
        IDs, Es, Es_ = [], [], []
        SAE = 0  # Sum absolute error.
        start = timeit.default_timer()

        for i, data in enumerate(dataloader):
            idx, E, E_ = self.model.forward(data)
            SAE_batch = torch.sum(torch.abs(E - E_), 0)
            SAE += SAE_batch
            IDs += list(idx)
            Es += E.tolist()
            Es_ += E_.tolist()

            if (time is True and i == 0):
                time = timeit.default_timer() - start
                minutes = len(dataloader) * time / 60
                hours = int(minutes / 60)
                minutes = int(minutes - 60 * hours)
                print('The prediction will finish in about',
                      hours, 'hours', minutes, 'minutes.')

        MAE = (SAE/N).tolist()  # Mean absolute error.
        MAE = ','.join([str(m) for m in MAE])  # For homo and lumo.

        prediction = 'ID\tCorrect\tPredict\tError\n'
        for idx, E, E_ in zip(IDs, Es, Es_):
            error = np.abs(np.array(E) - np.array(E_))
            error = ','.join([str(e) for e in error])
            E = ','.join([str(e) for e in E])
            E_ = ','.join([str(e) for e in E_])
            prediction += '\t'.join([idx, E, E_, error]) + '\n'

        return MAE, prediction

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

    def save_prediction(self, prediction, filename):
        with open(filename, 'w') as f:
            f.write(prediction)

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        self.directory = directory
        paths = sorted(Path(self.directory).iterdir(), key=os.path.getmtime)
        self.files = [str(p).strip().split('/')[-1] for p in paths]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return np.load(self.directory + self.files[idx], allow_pickle=True)


def mydataloader(dataset, batch_size, num_workers, shuffle=False):
    dataloader = torch.utils.data.DataLoader(
                 dataset, batch_size, shuffle=shuffle, num_workers=num_workers,
                 collate_fn=lambda xs: list(zip(*xs)), pin_memory=False)
    return dataloader


if __name__ == "__main__":

    """Args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('basis_set')
    parser.add_argument('radius')
    parser.add_argument('grid_interval')
    parser.add_argument('dim', type=int)
    parser.add_argument('layer_functional', type=int)
    parser.add_argument('hidden_HK', type=int)
    parser.add_argument('layer_HK', type=int)
    parser.add_argument('operation')
    parser.add_argument('batch_size', type=int)
    parser.add_argument('lr', type=float)
    parser.add_argument('lr_decay', type=float)
    parser.add_argument('step_size', type=int)
    parser.add_argument('iteration', type=int)
    parser.add_argument('setting')
    parser.add_argument('num_workers', type=int)
    
    parser.add_argument('lambdaV', type=float, default=1.0, help='weight of lossV')
    parser.add_argument('lambdaD', type=float, default=1.0, help='weight of lossD')
    parser.add_argument('--exch_scale', dest="exch_scale", type=float, default=1.0, help='weight of En')
    parser.add_argument('--use_d', dest='use_d', action='store_true', help='whether use the target D')
    parser.add_argument('--density_only', dest='density_only', action='store_true', help='QDF only use density')

    parser.add_argument('--eprefix', dest='eprefix', type=str, default="cur_exp", help="event folder prefix")
    parser.add_argument('--load_path', dest='load_path', type=str, default="", help="if the training process is two-step, this var specify the first step model path")
    parser.add_argument('--orbit_lrdecay', dest="orbit_lrdecay", type=float, default=1.0, help='lr decay of orbital parameters')

    args = parser.parse_args()
    dataset = args.dataset
    unit = '(' + dataset.split('_')[-1] + ')'
    basis_set = args.basis_set
    radius = args.radius
    grid_interval = args.grid_interval
    dim = args.dim
    layer_functional = args.layer_functional
    hidden_HK = args.hidden_HK
    layer_HK = args.layer_HK
    operation = args.operation
    batch_size = args.batch_size
    lr = args.lr
    lr_decay = args.lr_decay
    step_size = args.step_size
    iteration = args.iteration
    setting = args.setting
    num_workers = args.num_workers

    """Fix the random seed (with the taxicab number)."""
    torch.manual_seed(1729)

    
    """GPU or CPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU.')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU.')
    print('-'*50)

    """Create the dataloaders of training, val, and test set."""
    dir_dataset = '../dataset/' + dataset + '/'
    field = '_'.join([basis_set, radius + 'sphere', grid_interval + 'grid/'])
    # dataset_train = MyDataset(dir_dataset + 'train_' + field)
    # dataset_val = MyDataset(dir_dataset + 'val_' + field)
    # dataset_test = MyDataset(dir_dataset + 'test_' + field)

    dataset_train = MyDataset('/data/liuao/deep-DFT/QuantumDeepField_molecule-main/train/under15_train/')
    dataset_val = MyDataset('/data/liuao/deep-DFT/QuantumDeepField_molecule-main/train/under15_valid/')
    dataset_test = MyDataset('/data/liuao/deep-DFT/QuantumDeepField_molecule-main/train/under15_test/')

    dataloader_train = mydataloader(dataset_train, batch_size, num_workers,
                                    shuffle=True)
    dataloader_val = mydataloader(dataset_val, batch_size, num_workers)
    dataloader_test = mydataloader(dataset_test, batch_size, num_workers)
    print('# of training samples: ', len(dataset_train))
    print('# of validation samples: ', len(dataset_val))
    print('# of test samples: ', len(dataset_test))
    print('-'*50)

    """Load orbital_dict generated in preprocessing."""
    with open(dir_dataset + 'orbitaldict_' + basis_set + '.pickle', 'rb') as f:
        orbital_dict = pickle.load(f)
    N_orbitals = len(orbital_dict)

    """The output dimension in regression.
    When we learn only the atomization energy, N_output=1;
    when we learn the HOMO and LUMO simultaneously, N_output=2.
    """
    N_output = len(dataset_test[0][-2][0])

    print('Set a QDF model.')
    model = QuantumDeepField(device, N_orbitals,
                             dim, layer_functional, operation, N_output,
                             hidden_HK, layer_HK, density_only=args.density_only, use_d=args.use_d, exch_scale=args.exch_scale).to(device)
    trainer = Trainer(model, lr, lr_decay, step_size, lambdaV=args.lambdaV, use_d=args.use_d, lambdaD=args.lambdaD, orbit_lrdecay=args.orbit_lrdecay, load_path=args.load_path)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*50)

    """Output files."""
    file_result = "result_log.txt"
    result = ('Epoch\tTime(sec)\tLoss_E\tLoss_V\t'
              'MAE_val' + unit + '\tMAE_test' + unit)
    with open(file_result, 'w') as f:
        f.write(result + '\n')
    file_prediction = 'prediction.txt'
    file_model = 'model.pth'



    print('Start training of the QDF model with', dataset, 'dataset.\n'
          'The training result is displayed in this terminal every epoch.\n'
          'The result, prediction, and trained model '
          'are saved in the output directory.\n'
          'Wait for a while...')

    start = timeit.default_timer()


    tb_logger = SummaryWriter('{}_events'.format(args.eprefix))

    for epoch in range(iteration):
        loss_E, loss_V = trainer.train(dataloader_train)
        MAE_val = tester.test(dataloader_val)[0]
        MAE_test, prediction = tester.test(dataloader_test)
        time = timeit.default_timer() - start

        if epoch == 0:
            minutes = iteration * time / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-'*50)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_E, loss_V,
                                     MAE_val, MAE_test]))
        
        tb_logger.add_scalar('loss_E', loss_E, epoch)
        tb_logger.add_scalar('loss_V', loss_V, epoch)
        tb_logger.add_scalar('MAE_val', float(MAE_val), epoch)
        tb_logger.add_scalar('MAE_test', float(MAE_test), epoch)

        tester.save_result(result, file_result)
        tester.save_prediction(prediction, file_prediction)
        tester.save_model(model, file_model)
        print(result)

    print('The training has finished.')