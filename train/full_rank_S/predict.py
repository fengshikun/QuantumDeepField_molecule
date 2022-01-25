#!/usr/bin/env python3

import argparse
import pickle
import sys
import torch
import  os
dir = os.getcwd()
sys.path.append('../')
from train2 import *
print('dir',dir)
print(os.getcwd())

if __name__ == "__main__":

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

    parser.add_argument('--mha_num', dest="mha_num", type=int, default=2, help='mha layers number of coefficent network')

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

    mha_num = args.mha_num

    """GPU or CPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU.')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU.')
    print('-'*50)

    dir_dataset = '../../dataset/' + dataset + '/'
    # dir_trained = '../dataset/' + dataset_trained + '/'
    # dir_predict = '../dataset/' + dataset_predict + '/'
    dataset_predict = 'predict'
    # field = '_'.join([basis_set, radius + 'sphere', grid_interval + 'grid/'])
    #dataset_test = train.MyDataset(dir_predict + 'test_' + field)
    dataset_test = MyDataset('/data2/skfeng/QuantumDeepField_molecule/train/data/' + dataset_predict + '/')
    dataloader_test = mydataloader(dataset_test, batch_size=batch_size,
                                         num_workers=num_workers)

    with open(dir_dataset + 'orbitaldict_' + basis_set + '.pickle', 'rb') as f:
        orbital_dict = pickle.load(f)
    N_orbitals = len(orbital_dict)

    N_output = len(dataset_test[0][-2][0])

    model = QuantumDeepField(device, N_orbitals,
                             dim, layer_functional, operation, N_output, float(grid_interval),
                             hidden_HK, layer_HK, density_only=args.density_only, use_d=args.use_d, exch_scale=args.exch_scale, en_scale=args.en_scale, mha_num=mha_num).to(device)
    tester = Tester(model)

    print('Start predicting for', dataset_predict, 'dataset.\n'          
          'The prediction result is saved in the prediction filefold.\n'
          'Wait for a while...')
    print('batch_size',batch_size)
    MAE, prediction_E, prediction_V, prediction_D = tester.test(dataloader_test, time=True,predict=True)
    isExists=os.path.exists(dir + '/prediction/')
    if not isExists:
        os.makedirs(dir + '/prediction/')
    filename = (dir + '/prediction/energy_' + dataset_predict + '.txt')
    tester.save_prediction(prediction_E, filename)
    filename = (dir + '/prediction/Vtot_' + dataset_predict + '.txt')
    tester.save_prediction(prediction_V, filename)
    filename = (dir + '/prediction/density' + dataset_predict + '.txt')
    tester.save_prediction(prediction_D, filename)

    print('MAE:', MAE)

    print('The prediction has finished.')
