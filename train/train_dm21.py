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
import sys
from util2 import *
from tensorboardX import SummaryWriter


class QuantumDeepField(nn.Module):
    def __init__(self, device, N_orbitals,
                 dim, layer_functional, operation, N_output, grid_interval,
                 hidden_HK, layer_HK, use_d=False, density_only=False, exch_scale=1.0, en_scale=1.0):
        super(QuantumDeepField, self).__init__()

        """All learning parameters of the QDF model."""
        self.coefficient = nn.Embedding(N_orbitals, dim)
        self.zeta = nn.Embedding(N_orbitals, 1)  # Orbital exponent.
        nn.init.ones_(self.zeta.weight)  # Initialize each zeta with one.
        
        self.G_max = 840
        self.N_cut_max = 130
        self.N_el_max = 68

        self.W_pre_func = nn.Linear(self.N_el_max, dim)
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
        self.en_scale = en_scale
        self.grid_interval = grid_interval


        self.G_max = 840
        self.N_cut_max = 130
        self.N_el_max = 68

        # self.ceff_module = AttentNelectron(self.G_max, self.N_el_max)
        self.ceff_encoder = AttentNelectronMHA(self.G_max, self.N_el_max)

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
                     distance_matrices, quantum_numbers, laplace=False, all=False):
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

        denom = GTOs.norm(2, 0, True).clamp_min(1e-12).expand_as(GTOs) * grid_interval**(3/2)
        n_GTOs = GTOs/denom
        
        if laplace == False:
            return n_GTOs
        else:
            """A laplace Gaussian-type orbital (GTO)        """
            g_GTOs = distance_matrices**(quantum_numbers-2) * torch.exp(-(zetas*distance_matrices)**2) * (-2*zetas**2*distance_matrices**2 +quantum_numbers-1)
            
            n_g_GTOs = g_GTOs/denom
#            -g_GTOs.dot(GTOs)/denom**3*GTOs
            
            
            l_GTOs = (distance_matrices**(quantum_numbers-3) *
                torch.exp(-zetas**2*distance_matrices**2)) * (4*zetas**4*distance_matrices**4
                -2*zetas**2*(2*quantum_numbers+1)*distance_matrices**2
                       +quantum_numbers*(quantum_numbers-1))
#            l_GTOs = (distance_matrices**(quantum_numbers-3) *
#                torch.exp(-zetas**2*distance_matrices**2)) * (4*zetas**4*distance_matrices**4
#                -2*zetas**2*(2*quantum_numbers-1)*distance_matrices**2
#                       +(quantum_numbers-2)*(quantum_numbers-1))
            n_l_GTOs=l_GTOs/denom
            
            
            if all:
                return n_GTOs,n_g_GTOs, n_l_GTOs
            else:
                return n_l_GTOs


    def LCAO2(self, inputs):
        """Inputs."""
        (atomic_orbitals, distance_matrices,
         quantum_numbers, N_electrons, N_fields) = inputs
        # print(distance_matrices[0].shape,distance_matrices[1].shape)
        # sys.exit()
        quantum_numbers_lst = []
        N_electrons_lst = []
        for i, ele in enumerate(quantum_numbers):
            N_electrons_lst.append(N_electrons[i][0][0])
            quantum_numbers_lst.append(ele.shape[1])

        """Cat or pad each input data for batch processing."""
        atomic_orbitals = self.list_to_batch(atomic_orbitals, torch.LongTensor)
        distance_matrices = self.pad(distance_matrices, 1e6)
        quantum_numbers = self.list_to_batch(quantum_numbers, cat=True, axis=1)
        N_electrons = self.list_to_batch(N_electrons)

        
        atomic_orbitals = torch.cat(atomic_orbitals)

        """LCAO."""
        basis_matrix_all, g_basis_matrix_all, l_basis_matrix_all = self.basis_matrix(atomic_orbitals,
                                         distance_matrices, quantum_numbers, laplace=True, all=True)

        
        
        # Todo get coefficients from network
#        """Normalize the coefficients in LCAO.可以去掉？"""
#        coefficients = []
#        for AOs in atomic_orbitals:
#            coefs = F.normalize(self.coefficient(AOs), 2, 0)
#            # print(torch.sum(torch.t(coefs)[0]**2))  # Normalization check.
#            coefficients.append(coefs)
#        coefficients = torch.cat(coefficients)
        # todo sample wise (coefficients is different per sample)
        
        # split the matrix
        # basis_matrix_lst = torch.split(basis_matrix_all, N_fields)
        # l_basis_matrix_list = torch.split(l_basis_matrix_all, N_fields)


        basis_matrix_lst = split_matrix(basis_matrix_all, N_fields, quantum_numbers_lst)
        l_basis_matrix_list = split_matrix(l_basis_matrix_all, N_fields, quantum_numbers_lst)
        g_basis_matrix_list = split_matrix(g_basis_matrix_all, N_fields, quantum_numbers_lst)
        

        # padding to same dimension
        coeffi_input = torch.zeros((len(basis_matrix_lst), self.N_cut_max, self.G_max)).cuda()
        for i, basis_matrix in enumerate(basis_matrix_lst):
            basis_matrix_t = torch.transpose(basis_matrix, 0, 1) # Nc X G
            coeffi_input[i, : basis_matrix_t.shape[0], :basis_matrix_t.shape[1]] = basis_matrix_t


        psi_lst = []
        psi_lap_lst = []
        psi_cpl_lst = []
        psi_cpl_lap_lst = []
        psi_gra_lst = []

        coeffi_batch = self.ceff_encoder(coeffi_input)

        for i, basis_matrix in enumerate(basis_matrix_lst):
            basis_matrix_t = torch.transpose(basis_matrix, 0, 1) # Nc X G
            l_basis_matrix_t = torch.transpose(l_basis_matrix_list[i], 0, 1) # Nc X G
            g_basis_matrix_t = torch.transpose(g_basis_matrix_list[i], 0, 1) # Nc X G
            coeffi_matrix = coeffi_batch[i,:basis_matrix_t.shape[0], :int(N_electrons_lst[i])].t()
            
            psi, psi_gra, psi_lap = get_orbital(coeffi_matrix, basis_matrix_t, g_basis_matrix_t, l_basis_matrix_t, grid_interval)
            psi_cpl, psi_cpl_lap = get_complement_orbital(basis_matrix_t, l_basis_matrix_t, psi, psi_lap, grid_interval)

            psi_lst.append(psi)
            psi_gra_lst.append(psi_gra)
            psi_lap_lst.append(psi_lap)
            psi_cpl_lst.append(psi_cpl)
            psi_cpl_lap_lst.append(psi_cpl_lap)

        return psi_lst,  psi_gra_lst, psi_lap_lst, psi_cpl_lst, psi_cpl_lap_lst

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

    def functional(self, vectors, layers):
        """DNN-based energy functional."""

        vectors = torch.relu(self.W_pre_func(vectors))
        for l in range(layers):
            vectors = torch.relu(self.W_functional[l](vectors))
        # if operation == 'sum':
        #     vectors = [torch.sum(vs, 0) for vs in torch.split(vectors, axis)]
        # if operation == 'mean':
        #     vectors = [torch.mean(vs, 0) for vs in torch.split(vectors, axis)]
        # return torch.stack(vectors)
        return self.W_property(vectors)

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
                # E = self.list_to_batch(data[6], cat=True, axis=0)  # Correct E.
                # V_n = self.list_to_batch(data[7], cat=True, axis=0)
                V_n_lst = data[7]
                psi_lst, psi_gra_lst, psi_lap_lst, psi_cpl_lst, psi_cpl_lap_lst = self.LCAO2(inputs)

                # way1 psi --> Vxch (Network)
                # for network propagation we need pad the N_el and transpose
                v_input_lst = []
                for ps in psi_lst:
                    pst = ps.transpose(0, 1)
                    pst_pad = padding_matrix(pst, self.N_el_max)
                    v_input_lst.append(pst_pad)
                v_input = torch.cat(v_input_lst, dim=0)

                V_xcH = self.functional(v_input,
                                            self.layer_functional) # G X 1
                # way2 get density rou
                densities_lst = []
                for psi in psi_lst: # Nel x G
                    psi_t = psi.transpose(0, 1) # G x Nel
                    dt = torch.sum(psi_t ** 2, 1)
                    dt = torch.unsqueeze(dt, 1)
                    densities_lst.append(dt)
                densities = torch.cat(densities_lst)
                

                # todo sample wise
                batch_num = len(data[0])
                K_lst = []
                K_cpl_lst = []
                E_k_lst = torch.zeros(batch_num).cuda()
                for i, psi in enumerate(psi_lst):
                    psi_lap = psi_lap_lst[i]
                    psi_cpl = psi_cpl_lst[i]
                    psi_cpl_lap = psi_cpl_lap_lst[i]
                    # way3 psi, psi_lap --> K
                    K = -0.5 * torch.matmul(psi, psi_lap.transpose(0, 1)) * grid_interval**3
                    # way4 psi_cpl, psi_cpl_lap --> K_cpl
                    K_cpl = -0.5 * torch.matmul(psi_cpl, psi_cpl_lap.transpose(0, 1)) * grid_interval**3
                    K_lst.append(K)
                    # K_lst[i] = K
                    # K_cpl_lst[i] = K_cpl
                    K_cpl_lst.append(K_cpl)
                    # get Ek from K
                    E_k = torch.sum(torch.diagonal(K))
                    # E_k_lst.append(E_k)
                    E_k_lst[i] = E_k

                # get En from (Vn, rou), Exch from (Vxch, rou)
                E_n = torch.zeros(batch_num).cuda()
                E_xch = torch.zeros(batch_num).cuda()
                # V_xcH_lst = torch.split(V_xcH, N_fields) 
                d_n = 0
                V_xcH_lst = []   
                for i in range(batch_num):
                    d_ni=data[2][i].shape[0]
                    E_n[i] = torch.sum(torch.from_numpy(V_n_lst[i]).cuda() * densities[d_n:d_n+d_ni])  * grid_interval**3
                    V_xcH_lst.append(V_xcH[d_n:d_n + d_ni])
                    E_xch[i] = -torch.sum(V_xcH[d_n:d_n + d_ni] * densities[d_n:d_n + d_ni]) * grid_interval**3
                    d_n += d_ni
                
                E_ = self.exch_scale * E_xch + self.en_scale * E_n + E_k_lst
                E_ = torch.unsqueeze(E_, 1)
            return idx, E_

        elif train:
            # with torch.autograd.set_detect_anomaly(True):
            statistics_dict = {}
            E = self.list_to_batch(data[6], cat=True, axis=0)  # Correct E.
            # V_n = self.list_to_batch(data[7], cat=True, axis=0)
            V_n_lst = data[7]
            psi_lst, psi_gra_lst, psi_lap_lst, psi_cpl_lst, psi_cpl_lap_lst = self.LCAO2(inputs)

            # way1 psi --> Vxch (Network)
            # for network propagation we need pad the N_el and transpose
            v_input_lst = []
            for ps in psi_lst:
                pst = ps.transpose(0, 1)
                pst_pad = padding_matrix(pst, self.N_el_max)
                v_input_lst.append(pst_pad)
            v_input = torch.cat(v_input_lst, dim=0)

            V_xcH = self.functional(v_input,
                                        self.layer_functional) # G X 1
            #prepare dm21 input
            # way2 get density rou
            densities_lst = []
            for psi in psi_lst: # Nel x G
                psi_t = psi.transpose(0, 1) # G x Nel #是否有必要转置？
                dt = torch.sum(psi_t ** 2, 1)
                dt = torch.unsqueeze(dt, 1)
                densities_lst.append(dt)
            densities = torch.cat(densities_lst)
            
            #gradient of density
#            densities_gra_lst = []
#            for psi_gra in psi_gra_lst: # Nel x G
#                psi_t = psi.transpose(0, 1) # G x Nel #是否有必要转置？
#                dt = 2*torch.sum(psi_gra *psi_t, 1)#两个list元素相乘
#                dt = torch.unsqueeze(dt, 1)
#                densities_lst.append(dt)
#            densities_gra = torch.cat(densities_gra_lst)
       
            

            # todo sample wise
            batch_num = len(data[0])
            K_lst = []
            K_cpl_lst = []
            tau_lst = []
            densities_gra_lst = []
            E_k_lst = torch.zeros(batch_num).cuda()
            for i, psi in enumerate(psi_lst):
                psi_lap = psi_lap_lst[i]
                psi_gra = psi_gra_lst[i]
                psi_cpl = psi_cpl_lst[i]
                psi_cpl_lap = psi_cpl_lap_lst[i]
                
                # gradient of density(r)
                # print('shape', psi_gra.shape, psi.shape)
                densities_gra = 2 * torch.sum(torch.mul(psi,psi_gra), 0 )
                # print('shape', densities_gra.shape)
                densities_gra_lst.append(densities_gra)
                
                # kinetic density(r)
                tau = torch.sum(torch.mul(psi,psi_lap), 0 )            
#                tau = torch.unsqueeze(tau, 0)
                tau_lst.append(tau)
                
                #check boundary condition  psi:Nel*G
                # print('check1', torch.sum(tau)* grid_interval**3)
                # print('check2', -torch.sum(torch.mul(psi, psi_gra)* grid_interval**3,1))
                # print('check3', torch.sum(torch.mul(psi, psi)* grid_interval**3,1))
                # sys.exit()
                # way3 psi, psi_lap --> K
                K = -0.5 * torch.matmul(psi, psi_lap.transpose(0, 1)) * grid_interval**3
                # way4 psi_cpl, psi_cpl_lap --> K_cpl
                K_cpl = -0.5 * torch.matmul(psi_cpl, psi_cpl_lap.transpose(0, 1)) * grid_interval**3
                K_lst.append(K)
                # K_lst[i] = K
                # K_cpl_lst[i] = K_cpl
                K_cpl_lst.append(K_cpl)
                # get Ek from K
                E_k = torch.sum(torch.diagonal(K))
                # E_k_lst.append(E_k)
                E_k_lst[i] = E_k

            # get En from (Vn, rou), Exch from (Vxch, rou)
            E_n = torch.zeros(batch_num).cuda()
            E_xch = torch.zeros(batch_num).cuda()
#            V_xcH_lst = torch.split(V_xcH, N_fields) 
            d_n = 0
            V_xcH_lst = []   
            for i in range(batch_num):
                d_ni=data[2][i].shape[0]
                E_n[i] = torch.sum(torch.from_numpy(V_n_lst[i]).cuda() * densities[d_n:d_n+d_ni])  * grid_interval**3
                V_xcH_lst.append(V_xcH[d_n:d_n + d_ni])
                E_xch[i] = -torch.sum(V_xcH[d_n:d_n + d_ni] * densities[d_n:d_n + d_ni]) * grid_interval**3
                d_n += d_ni
            
            E_ = self.exch_scale * E_xch + self.en_scale * E_n + E_k_lst

            statistics_dict = {
                "E_xcH": torch.mean(E_xch),
                "En": torch.mean(E_n),
                "Ek": torch.mean(E_k_lst)
            }
            loss1 = F.smooth_l1_loss(E, torch.unsqueeze(E_, 1))

            # loss2
            diag_loss = 0
            gt_con_loss = 0
            for i, V_n in enumerate(V_n_lst):
                V_total = torch.from_numpy(V_n).cuda() + V_xcH_lst[i]
                V = getV(psi_lst[i], V_total)
                V_cpl = getV(psi_cpl_lst[i], V_total)
                diag_loss += diag_constraint_loss(V, K_lst[i])
                gt_con_loss += get_constraint_loss(V, K_lst[i], V_cpl, K_cpl_lst[i])
            diag_loss /= batch_num
            gt_con_loss /= batch_num

            return loss1, diag_loss, gt_con_loss, statistics_dict

        else:  # Test.
            with torch.no_grad():
                E = self.list_to_batch(data[6], cat=True, axis=0)  # Correct E.
                # V_n = self.list_to_batch(data[7], cat=True, axis=0)
                V_n_lst = data[7]
                psi_lst, psi_gra_lst, psi_lap_lst, psi_cpl_lst, psi_cpl_lap_lst = self.LCAO2(inputs)

                # way1 psi --> Vxch (Network)
                # for network propagation we need pad the N_el and transpose
                v_input_lst = []
                for ps in psi_lst:
                    pst = ps.transpose(0, 1)
                    pst_pad = padding_matrix(pst, self.N_el_max)
                    v_input_lst.append(pst_pad)
                v_input = torch.cat(v_input_lst, dim=0)

                V_xcH = self.functional(v_input,
                                            self.layer_functional) # G X 1
                # way2 get density rou
                densities_lst = []
                for psi in psi_lst: # Nel x G
                    psi_t = psi.transpose(0, 1) # G x Nel
                    dt = torch.sum(psi_t ** 2, 1)
                    dt = torch.unsqueeze(dt, 1)
                    densities_lst.append(dt)
                densities = torch.cat(densities_lst)
                

                # todo sample wise
                batch_num = len(data[0])
                K_lst = []
                K_cpl_lst = []
                E_k_lst = torch.zeros(batch_num).cuda()
                for i, psi in enumerate(psi_lst):
                    psi_lap = psi_lap_lst[i]
                    psi_cpl = psi_cpl_lst[i]
                    psi_cpl_lap = psi_cpl_lap_lst[i]
                    # way3 psi, psi_lap --> K
                    K = -0.5 * torch.matmul(psi, psi_lap.transpose(0, 1)) * grid_interval**3
                    # way4 psi_cpl, psi_cpl_lap --> K_cpl
                    K_cpl = -0.5 * torch.matmul(psi_cpl, psi_cpl_lap.transpose(0, 1)) * grid_interval**3
                    K_lst.append(K)
                    # K_lst[i] = K
                    # K_cpl_lst[i] = K_cpl
                    K_cpl_lst.append(K_cpl)
                    # get Ek from K
                    E_k = torch.sum(torch.diagonal(K))
                    # E_k_lst.append(E_k)
                    E_k_lst[i] = E_k

                # get En from (Vn, rou), Exch from (Vxch, rou)
                E_n = torch.zeros(batch_num).cuda()
                E_xch = torch.zeros(batch_num).cuda()
                # V_xcH_lst = torch.split(V_xcH, N_fields) 
                d_n = 0
                V_xcH_lst = []   
                for i in range(batch_num):
                    d_ni=data[2][i].shape[0]
                    E_n[i] = torch.sum(torch.from_numpy(V_n_lst[i]).cuda() * densities[d_n:d_n+d_ni])  * grid_interval**3
                    V_xcH_lst.append(V_xcH[d_n:d_n + d_ni])
                    E_xch[i] = -torch.sum(V_xcH[d_n:d_n + d_ni] * densities[d_n:d_n + d_ni]) * grid_interval**3
                    d_n += d_ni
                
                E_ = self.exch_scale * E_xch + self.en_scale * E_n + E_k_lst
                E_ = torch.unsqueeze(E_, 1)
            return idx, E, E_


class Trainer(object):
    def __init__(self, model, lr, lr_decay, step_size, use_d=False, orbit_lrdecay=1, load_path=""):
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
            # for name, p in model.named_parameters():
            #     if name.startswith("coefficient") or name.startswith("zeta"):
            #         p.requires_grad = False
            #         print("Freeze the weight of {}".format(name))
        
            self.optimizer = optim.Adam(rest_params, lr)
        else:
            self.optimizer = optim.Adam([{'params':rest_params,'lr':lr},
                        {'params':model.coefficient.parameters(),'lr':lr * orbit_lrdecay},
                        {'params':model.zeta.parameters(),'lr':lr * orbit_lrdecay}
                        ])

        # self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size, lr_decay)

        self.use_d = use_d

    def optimize(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, dataloader, epoch_num, lossV_weight = 0.00001):
        """Minimize two loss functions in terms of E and V."""
        losses_E, losses_diag, losses_gt = 0, 0, 0
        statistics_dict = {}
        cnt = 1
        print("lossV weight is {}".format(lossV_weight))
        for data in dataloader:
            # with torch.autograd.set_detect_anomaly(True):
            loss_E, diag_loss, gt_con_loss, statistics_dict = self.model.forward(data, train=True, target='D')
            total_loss = loss_E + lossV_weight * (diag_loss +  gt_con_loss)
            self.optimize(total_loss, self.optimizer)
            losses_E += loss_E.item()
            losses_diag += diag_loss.item()
            losses_gt += gt_con_loss.item()
            result = '\t'.join(map(str, [epoch, cnt, len(dataloader), loss_E.item(), diag_loss.item(), gt_con_loss.item()]))
            print(result)
            cnt += 1

            tb_logger.add_scalar('loss_E', loss_E, epoch)
            tb_logger.add_scalar('losses_diag', diag_loss, epoch)
            tb_logger.add_scalar('losses_gt', gt_con_loss, epoch)
            if len(statistics_dict):
                for k in statistics_dict:
                    if k in ["zeta_sq", "min_coeffe", "max_coeffe", "med_coeffe"]:
                        tb_logger.add_histogram(k, statistics_dict[k], epoch)
                    else:
                        tb_logger.add_scalar(k, statistics_dict[k], epoch)
            
            # break

        self.scheduler.step()

        return losses_E, losses_diag, losses_gt



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
    parser.add_argument('--exch_scale', dest="exch_scale", type=float, default=1.0, help='weight of Exch')
    parser.add_argument('--en_scale', dest="en_scale", type=float, default=1.0, help='weight of En')
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
    grid_interval = float(args.grid_interval)
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
    # dir_dataset = '../../dataset/' + dataset + '/'
    dir_dataset = '/data2/skfeng/QuantumDeepField_molecule/dataset/' + dataset + '/'
    # field = '_'.join([basis_set, radius + 'sphere', grid_interval + 'grid/'])
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
                             dim, layer_functional, operation, N_output, float(grid_interval),
                             hidden_HK, layer_HK, density_only=args.density_only, use_d=args.use_d, exch_scale=args.exch_scale, en_scale=args.en_scale).to(device)
    trainer = Trainer(model, lr, lr_decay, step_size, use_d=args.use_d, orbit_lrdecay=args.orbit_lrdecay, load_path=args.load_path)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*50)

    """Output files."""
    file_result = "result_log.txt"
    result = ('Epoch\tTime(sec)\tLoss_E\tLoss_diag\tLoss_gt\t'
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
        loss_E, losses_diag, losses_gt = trainer.train(dataloader_train, epoch, args.lambdaV)
        MAE_val = tester.test(dataloader_val)[0]
        MAE_test, prediction = tester.test(dataloader_test)
        time = timeit.default_timer() - start

        tb_logger.add_scalar('MAE_val', float(MAE_val), epoch)
        tb_logger.add_scalar('MAE_test', float(MAE_test), epoch)

        if epoch == 0:
            minutes = iteration * time / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-'*50)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_E, losses_diag, losses_gt,
                                     MAE_val, MAE_test]))
        print(result)
        

        tester.save_result(result, file_result)
        tester.save_prediction(prediction, file_prediction)
        tester.save_model(model, file_model)
        

    print('The training has finished.')