import time
import argparse
import sys
sys.path.append('/path/to/MVP')
from utils import get_logger
from training.trainer_img import ICMVtrainer
import os
from statistics import mean

def get_trainer_from_args(args):

    args.seed = [42]
    args.weight = [5, 2.5] # Z, omega

    if args.dataset == 'MMNIST':
        args.data_path = '/path/to/MVP/data/MMNIST'
        args.num_views = 5
        args.num_classes = 10
        args.class_dim = 48
        args.latent_dim = 96
        args.image_shape = [3, 28, 28]
        args.arch = None
        args.transform = [256, 1024, 256]
        args.batch_size = 256
        args.lr = 0.0005
        args.normalizing_factor = [2,1]
        args.epochs = 500 
        args.interval = 100
        args.likelihood = 'Laplace'

    elif args.dataset == 'MVShapeNet':
        args.data_path = '/path/to/MVP/data/MVShapeNet'
        args.num_views = 5
        args.num_classes = 5
        args.class_dim = 192
        args.latent_dim = 256
        args.image_shape = [3, 64, 64]
        args.arch = None
        args.transform = [256, 512, 256] 
        args.batch_size = 256
        args.lr = 0.0001
        args.normalizing_factor = [2,1]
        args.epochs = 500
        args.interval = 100
        args.likelihood = 'Normal'


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
    parser.add_argument('--trainer_mode', type=str, default='cluster', metavar='N',
                    help='Training task: class/cluster')
    parser.add_argument('--dataset', type=str, default='Scene15', metavar='N',
                    help='Dataset name')         
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
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
    parser.add_argument('--image_shape', default=[3, 28, 28], nargs='+',
                    help='input image shape') 
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
    parser.add_argument('--class_dim', type=int, default=10, metavar='L',
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
