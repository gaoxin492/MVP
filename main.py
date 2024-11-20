import time
import argparse
import sys
sys.path.append('/path/to/MVP')
from utils import get_logger
from training.trainer import ICMVtrainer
import os
from statistics import mean

def get_trainer_from_args(args):

    args.lr = 0.001
    args.seed = [42, 44, 46, 48, 50]
    args.weight = [5, 2.5] # Z, omega

    if args.dataset == 'Handwritten':
        args.data_path = '/path/to/MVP/data/Handwritten'
        args.num_views = 6
        args.in_channels = [76, 216, 64, 47, 240, 6]
        args.arch = dict(
                            m0=[76,  256, 256, 1024],
                            m1=[216, 512, 512, 1024],
                            m2=[64,  256, 256, 1024],
                            m3=[47,  256, 256, 1024],
                            m4=[240, 512, 512, 1024],
                            m5=[6,   256, 256, 512]
                            )
        args.num_classes = 10
        args.latent_dim = 16
        args.epochs = 150
        args.likelihood = 'Laplace'
        args.normalizing_factor = [2.5, 2]


    elif args.dataset == 'CUB':
        args.data_path = '/path/to/MVP/data/CUB'
        args.num_views = 2
        args.in_channels = [1024, 300]
        args.arch = dict(
                            m0=[1024, 256, 256, 1024],
                            m1=[300,  256, 256, 1024],
                            )
        args.num_classes = 10
        args.latent_dim = 16
        args.epochs = 150
        args.likelihood = 'Normal'
        args.normalizing_factor = [1, 1]


    elif args.dataset == 'Scene15':
        args.data_path = '/path/to/MVP/data/Scene15'
        args.num_views = 3
        args.in_channels = [20, 59, 40]
        args.arch = dict(
                            m0=[20, 512, 512, 2048], 
                            m1=[59, 512, 512, 2048],
                            m2=[40, 512, 512, 2048],
                            )
        args.num_classes = 15
        args.latent_dim = 16
        args.epochs = 250 
        args.likelihood = 'Normal'
        args.normalizing_factor = [10, 1]
    

    elif args.dataset == 'Reuters10':
        args.data_path = '/path/to/MVP/data/Reuters10'
        args.num_views = 5
        args.in_channels = [10, 10, 10, 10, 10]
        args.arch = dict(
                            m0=[10, 512, 512, 2048],
                            m1=[10, 512, 512, 2048],
                            m2=[10, 512, 512, 2048],
                            m3=[10, 512, 512, 2048],
                            m4=[10, 512, 512, 2048],
                            )
        args.num_classes = 6
        args.latent_dim = 16
        args.epochs = 60 
        args.likelihood = 'Normal'
        args.normalizing_factor = [5, 1]
    

    elif args.dataset == 'SensIT_Vehicle':
        args.data_path = '/path/to/MVP/data/SensIT_Vehicle'
        args.num_views = 2
        args.in_channels = [50, 50]
        args.arch = dict(
                            m0=[50, 1024, 1024, 2048],
                            m1=[50, 1024, 1024, 2048],
                            )
        args.num_classes = 3
        args.latent_dim = 32
        args.epochs = 50
        args.likelihood = 'Normal'
        args.normalizing_factor = [5, 1]


    args.class_dim = args.latent_dim
    args.interval = args.epochs
    trainer = ICMVtrainer(args)

    return trainer


def run_training(args):

    trainer = get_trainer_from_args(args)
    trainer.run_training()
    

def run_training_entry():

    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    parser = argparse.ArgumentParser()
    # Basic Information
    parser = argparse.ArgumentParser(description='Dir-VAE MNIST Example')
    parser.add_argument('--dataset', type=str, default='Scene15', metavar='N',
                    help='Dataset name')         
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)') #epoch was 10
    parser.add_argument('--warm_up', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)') #epoch was 10
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=['42'], nargs='+',
                    help='different seeds, length indicates test time') 
    # parser.add_argument('--data_seed', type=int, default=42, metavar='S',
    #                 help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--num_views', type=int, default=6, metavar='K',
                    help='the number of modalities in the dataset')   # You need to change
    parser.add_argument('--in_channels', default=None, nargs='+',
                    help='dimension of different views') 
    parser.add_argument('--transform', default=[128, 256, 128], nargs='+',
                    help='inter-view transformation')
    parser.add_argument('--num_classes', type=int, default=10, metavar='K',
                    help='the number of classes in the dataset')   # You need to change
    parser.add_argument('--weight', type=float, default=10, nargs='+',
                    help='the weight of the KL divergence term')   # You need to change
    parser.add_argument('--normalizing_factor', type=float, default=1, metavar='M')  
    parser.add_argument('--missing_rate', type=float, default=0.0, metavar='M',
                    help='the missing rate of the input data')   # You need to change
    parser.add_argument('--class_dim', type=int, default=16, metavar='L',
                    help='the dimension of the latent space')   # You need to change
    parser.add_argument('--latent_dim', type=int, default=16, metavar='L',
                    help='the dimension of the latent space')   # You need to change

    parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
    parser.add_argument('--log_path', default='/path/to/MVP/results', type=str) # You need to change
    parser.add_argument('--log_name', default='train_log', type=str)
    parser.add_argument('--data_path', default='/path/to/MVP/data/Handwritten', type=str)

    parser.add_argument('--lr', default=0.001, type=float, help="initial learning rate")  # You need to change
    parser.add_argument('--weight_decay', default=1e-5, type=float, help="weight decay")
    
    args = parser.parse_args()

    run_training(args)

if __name__ == '__main__':
    run_training_entry()
