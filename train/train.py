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
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class QuantumDeepField(nn.Module):
    def __init__(self, device, N_orbitals,
                 dim, layer_functional, operation, N_output,
                 hidden_HK, layer_HK):
        super(QuantumDeepField, self).__init__()

        """All learning parameters of the QDF model."""
        self.coefficient = nn.Embedding(N_orbitals, dim)
        self.zeta = nn.Embedding(N_orbitals, 1)  # Orbital exponent.
        nn.init.ones_(self.zeta.weight)  # Initialize each zeta with one.
        self.W_pre_func = nn.Linear(1, dim) #should not be like this!
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
                     distance_matrices, quantum_numbers, laplace):
        """Transform the distance matrix into a basis matrix,
        in which each element is d^(q-1)*e^(-z*d^2), where d is the distance,
        q is the principle quantum number, and z is the orbital exponent.
        We note that this is a simplified Gaussian-type orbital (GTO)
        in terms of the spherical harmonics.
        We simply normalize the GTO basis matrix using F.normalize in PyTorch.
        """
        zetas = torch.squeeze(self.zeta(atomic_orbitals))
        GTOs = (distance_matrices**(quantum_numbers-1) *
                torch.exp(-zetas*distance_matrices**2))
        denom = GTOs.norm(2, 0, True).clamp_min(1e-12).expand_as(GTOs)
        n_GTOs = GTOs/denom
        g_GTOs = (distance_matrices**(quantum_numbers-2) * torch.exp(-zetas*distance_matrices**2)) * (-2*zetas*distance_matrices**2 +quantum_numbers-1)
        g_GTOs = g_GTOs/denom
        if laplace == False:
            return n_GTOs, g_GTOs
        else:
            """A laplace Gaussian-type orbital (GTO)        """
            l_GTOs = (distance_matrices**(quantum_numbers-3) *
                torch.exp(-zetas*distance_matrices**2)) * (4*zetas**2*distance_matrices**4
                -2*zetas*(2*quantum_numbers+1)*distance_matrices**2
                       +quantum_numbers*(quantum_numbers-1)) #should not be like this!
            return l_GTOs/denom, g_GTOs
            
    def LCAO(self, inputs,laplace=True):
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
            coefficients.append(coefs)
        coefficients = torch.cat(coefficients)
        
        atomic_orbitals = torch.cat(atomic_orbitals)
        
        """LCAO."""
        basis_matrix, g_GTOs = self.basis_matrix(atomic_orbitals,
                                         distance_matrices, quantum_numbers,laplace = False)
        molecular_orbitals = torch.matmul(basis_matrix, coefficients)

        """We simply normalize the molecular orbitals
        and keep the total electrons of the molecule
        in learning the molecular orbitals.
        """
        split_MOs = torch.split(molecular_orbitals, N_fields)
        normalized_MOs = []
        denom_list = []
        for N_elec, MOs in zip(N_electrons, split_MOs):
            eps = 1e-12
            denom = MOs.norm(2, 0, True).clamp_min(eps).expand_as(MOs)
            n_MTOs = MOs / denom
            MOs = torch.sqrt(N_elec / self.dim) * n_MTOs
            denom_list.append(denom)
            normalized_MOs.append(MOs)
        if laplace == False:
            return torch.cat(normalized_MOs),1
        else :
            basis_matrix, g_basis_matrix = self.basis_matrix(atomic_orbitals,
                                         distance_matrices, quantum_numbers,laplace = True)
            molecular_orbitals = torch.matmul(basis_matrix, coefficients)
            split_MOs = torch.split(molecular_orbitals, N_fields)
            normalized_MOs = []
            index = 0
            for N_elec, MOs in zip(N_electrons, split_MOs):
                n_MTOs = MOs / denom_list[index]
                MOs = torch.sqrt(N_elec / self.dim) * n_MTOs
                normalized_MOs.append(MOs)
                index += 1
            g_molecular_orbitals = torch.matmul(g_basis_matrix, coefficients)
            g_split_MOs = torch.split(g_molecular_orbitals, N_fields)
            g_normalized_MOs = []
            index = 0
            for N_elec, MOs in zip(N_electrons, g_split_MOs):
                n_MTOs = MOs / denom_list[index]
                MOs = torch.sqrt(N_elec / self.dim) * n_MTOs
                g_normalized_MOs.append(MOs)
                index += 1

            return torch.cat(normalized_MOs), torch.cat(g_normalized_MOs)
            


    def functional(self, vectors, layers, operation, axis):
        """DNN-based energy functional."""
        #vectors = self.W_pre_func(vectors)
        for l in range(layers):
            vectors = torch.relu(self.W_functional[l](vectors))
        if operation == 'sum':
            vectors = [torch.sum(vs, 0) for vs in torch.split(vectors, axis)]
        if operation == 'mean':
            vectors = [torch.mean(vs, 0) for vs in torch.split(vectors, axis)]
        return torch.stack(vectors)

    def hxcmap(self, scalars, layers):
        """DNN-based map from desity to exchange-correlation potential."""
        vectors = self.W_density(scalars)
        for l in range(layers):
            vectors = torch.relu(self.W_HK[l](vectors))
        return self.W_potential(vectors)

    def forward(self, data, epsilon = [],  epoch = 0, train=False, target=None, predict=False):
        """Forward computation of the QDF model
        using the above defined functions.
        """

        idx, inputs, N_fields = data[0], data[1:6], data[5]

        if predict:  # For demo.
            with torch.no_grad():
                molecular_orbitals, temp = self.LCAO(inputs,laplace = False)
                final_layer = self.functional(molecular_orbitals,
                                              self.layer_functional,
                                              self.operation, N_fields)
                E_ = self.W_property(final_layer)
                return idx, E_

        elif train:
            E = self.list_to_batch(data[6], cat=True, axis=0)  # Correct E.
            l_molecular_orbitals, g_molecular_orbitals = self.LCAO(inputs,laplace = True)
            molecular_orbitals, temp = self.LCAO(inputs,laplace = False) 
            V_n = self.list_to_batch(data[7], cat=True, axis=0)  # Correct V.
            densities = torch.sum(molecular_orbitals ** 2, 1)
            densities = torch.unsqueeze(densities, 1)
            '''final_layer = self.functional(densities,
                                          self.layer_functional,
                                          self.operation, N_fields)'''
            final_layer = self.functional(molecular_orbitals,
                                        self.layer_functional,
                                        self.operation, N_fields)
            E_xcH = self.W_property(final_layer)
            grad_v = []
            batch_num = len(data[2])
            for i in range(batch_num):
                grad_v.append(torch.autograd.grad(outputs=E_xcH[i], inputs=molecular_orbitals, retain_graph=True)[0])
            V_xcH = grad_v[0]
            dim_num = 0
            for i in range(batch_num):
                V_xcH[dim_num:dim_num + data[2][i].shape[0]] = grad_v[i][dim_num:dim_num + data[2][i].shape[0]]
                dim_num += data[2][i].shape[0]
            V = V_xcH + V_n  # Predicted V.
#            temp_mul = torch.sum(2*molecular_orbitals * g_molecular_orbitals, 1).reshape(-1, 1)
#            E_n = V_n * temp_mul
            E_n = V_n * densities
            E_n = [torch.sum(vs, 0) for vs in torch.split(E_n, N_fields)]
            E_n = torch.stack(E_n)
            E_ = E_xcH + E_n
#            E_ = E_xcH
#            f = open('E_xcH.txt', 'a+')
#            f.write(str(E_xcH))
#            f.write('\n')
#            '''f = open('final_layer.txt', 'a+')
#            f.write(str(final_layer))
#            f.write('\n')'''
#            f = open('V_xcH.txt', 'a+')
#            f.write(str(V_xcH))
#            f.write('\n')
#            f = open('V_n.txt', 'a+')
#            f.write(str(V_n))
#            f.write('\n')
#            f = open('E_n.txt', 'a+')
#            f.write(str(E_n))
#            f.write('\n')
#            '''print("molecular.shape: ", molecular_orbitals.shape)
#            print("V_n.shape: ", V_n.shape)
#            print("densities.shape: ", densities.shape)
#            print("E_xch.shape: ", E_xcH.shape)
#            print("E.shape: ", E.shape)
#            print("V_xch.shape: ", V_xcH.shape)
#            print("E_n.shape: ", E_n.shape)
#            print("E.shape: ", E.shape)'''
            loss1 = F.mse_loss(E, E_)
            temp_loss = []
            dim_num = 0
            for i in range(batch_num):
                #I don't know why it occurs wrongs, it tell me RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation. Solve: I copy the epsilon[i], make them point to different address
                epis = copy.copy(epsilon[i])
                temp_op2 = molecular_orbitals[dim_num:dim_num + data[2][i].shape[0]].clone()*epis
                temp_loss.append(F.mse_loss(V[dim_num:dim_num + data[2][i].shape[0]].clone()*molecular_orbitals[dim_num:dim_num + data[2][i].shape[0]].clone()-1/2*l_molecular_orbitals[dim_num:dim_num + data[2][i].shape[0]].clone(),temp_op2))
                dim_num +=data[2][i].shape[0]
            loss2 = sum(temp_loss)
            if (epoch%2==1):
                #loss = loss1
                loss = loss1
            else:
                loss = loss2
#            loss = loss1 + alpha*loss2
#            loss = loss1
            return loss

        else:  # Test.
            with torch.no_grad():
                E = self.list_to_batch(data[6], cat=True, axis=0)
                molecular_orbitals, temp = self.LCAO(inputs, laplace = False)
                final_layer = self.functional(molecular_orbitals,
                                              self.layer_functional,
                                              self.operation, N_fields)
                E_ = self.W_property(final_layer)
                return idx, E, E_


class Trainer(object):
    def __init__(self, model, lr, lr_decay, step_size):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size, lr_decay)

    def optimize(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, dataloader, epoch):
        """Minimize two loss functions in terms of E and V."""
        losses = 0
        for data in dataloader:
            #1 we solve epsilon for each molecular, there are batch_size moleculars
            inputs = data[1:6]
            molecular_orbitals, temp = self.model.LCAO(inputs,laplace = False)
            V_n = self.model.list_to_batch(data[7], cat=True, axis=0)  # Correct V.
            densities = torch.sum(molecular_orbitals ** 2, 1)
            densities = torch.unsqueeze(densities, 1)
            V_xcH = self.model.hxcmap(densities, self.model.layer_HK) #should not be like this!
            V = V_xcH + V_n  
            l_molecular_orbitals, temp = self.model.LCAO(inputs, laplace = True)   
            batch_num = len(data[2])
            epsilon = []
            dim_num = 0
            for i in range(batch_num):
                num1 = ((V[dim_num:dim_num + data[2][i].shape[0]]*molecular_orbitals[dim_num:dim_num + data[2][i].shape[0]]-1/2*l_molecular_orbitals[dim_num:dim_num + data[2][i].shape[0]])*molecular_orbitals[dim_num:dim_num + data[2][i].shape[0]]).sum()
                num2 = (molecular_orbitals[dim_num:dim_num + data[2][i].shape[0]]*molecular_orbitals[dim_num:dim_num + data[2][i].shape[0]]).sum()
                dim_num += data[2][i].shape[0]
                epsilon.append(num1/num2)
            #2 for each molecular's loss, we have an epsilon.we sum them to the total loss and backward
            loss= self.model.forward(data, epsilon, epoch, train=True, target='E')
            self.optimize(loss, self.optimizer)
            losses += loss.item()
        self.scheduler.step()
        return losses


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
                 collate_fn=lambda xs: list(zip(*xs)), pin_memory=True)
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
                             hidden_HK, layer_HK).to(device)
    trainer = Trainer(model, lr, lr_decay, step_size)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*50)

    """Output files."""
    file_result = '../output/result--' + setting + '.txt'
    result = ('Epoch\tTime(sec)\tLoss_E\tLoss_V\t'
              'MAE_val' + unit + '\tMAE_test' + unit)
    with open(file_result, 'w') as f:
        f.write(result + '\n')
    file_prediction = '../output/prediction--' + setting + '.txt'
    file_model = '../output/model--' + setting

    print('Start training of the QDF model with', dataset, 'dataset.\n'
          'The training result is displayed in this terminal every epoch.\n'
          'The result, prediction, and trained model '
          'are saved in the output directory.\n'
          'Wait for a while...')

    start = timeit.default_timer()

    for epoch in range(iteration):
        loss= trainer.train(dataloader_train, epoch)
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

        result = '\t'.join(map(str, [epoch, time, loss,
                                     MAE_val, MAE_test]))
        f = open('loss.txt', 'a+')
        f.write(str(epoch))
        f.write('\n')
        f.write(str(loss))
        f.write('\n')

        #save result for each epoch
        f = open('MAE_val.txt', 'a+')
        f.write(str(MAE_val))
        f.write('\n')
        f = open('MAE_test.txt', 'a+')
        f.write(MAE_test)
        f.write('\n')

        tester.save_result(result, file_result)
        tester.save_prediction(prediction, file_prediction)
        tester.save_model(model, file_model)
        print(result)

    print('The training has finished.')
