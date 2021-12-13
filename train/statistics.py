import torch
from pathlib import Path
import os
import numpy as np

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

dataset_train = MyDataset('/data/liuao/deep-DFT/QuantumDeepField_molecule-main/train/under15_train/')
dataset_val = MyDataset('/data/liuao/deep-DFT/QuantumDeepField_molecule-main/train/under15_valid/')
dataset_test = MyDataset('/data/liuao/deep-DFT/QuantumDeepField_molecule-main/train/under15_test/')

batch_size = 8
num_workers = 2

dataloader_train = mydataloader(dataset_train, batch_size, num_workers,
                                shuffle=True)
dataloader_val = mydataloader(dataset_val, batch_size, num_workers)
dataloader_test = mydataloader(dataset_test, batch_size, num_workers)



def statistic_val(dls):
    N_fields_lst = [] # G
    quantum_numbers_lst = [] # N_cut
    N_electrons_lst = [] # N_elect    

    for dataloader in dls:
        for data in dataloader:
            idx, inputs, N_fields = data[0], data[1:6], data[5]
            (atomic_orbitals, distance_matrices,
                quantum_numbers, N_electrons, N_fields) = inputs
            # N_electrons_lst.append(np.concatenate(N_electrons))
            N_fields_lst.append(np.array(N_fields))
            for i, ele in enumerate(quantum_numbers):
                N_electrons_lst.append(N_electrons[i][0][0])
                quantum_numbers_lst.append(ele.shape[1])
                
    N_fields_lst = np.concatenate(N_fields_lst)
    quantum_numbers_lst = np.array(quantum_numbers_lst)
    N_electrons_lst = np.array(N_electrons_lst)
    print("Max G: {}")
            
statistic_val([dataloader_train, dataloader_val, dataloader_test])
            

