import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/path/to/MVP')
from model.vanilla_mvVAE_mnist import VanillaMVVAE
from dataprovider.MvImgDataset import generate_dataset
import random
from torchvision import datasets, transforms
import os
import PIL.Image as Image
from sklearn.metrics import accuracy_score
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from sklearn.cluster import KMeans
from sklearn import metrics
from tqdm import tqdm
from pretrained_classifier.ConvNetworkImgClfCMNIST import ClfImg as ClfImgCMNIST
import argparse
from statistics import mean
from collections import Counter


def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class Evaluator:
    def __init__(self, args):

        self.seed = args.data_seed
        self.model_path_dir = os.path.join('/path/to/MVP/results', args.model_path_dir)
        self.data_path = args.data_path
        self.missing_rate = args.missing_rate
        self.missing_view = args.missing_view
        self.num_views = 5
        self.num_classes = 10
        self.latent_dim = args.latent_dim
        self.class_dim = args.class_dim
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.log_file = os.path.join(self.model_path_dir, 'evaluate_log.txt')
        
        self.masks = torch.tensor([0 if i in args.missing_view else 1 for i in range(5)])
        
        self.setup_dataset()
        self.setup_classifiers()
        self.setup_model()
        self.kmeans = KMeans(n_clusters = 10, random_state=42, n_init='auto')

    def setup_model(self):
    
        self.img_save_path_dir = os.path.join(self.model_path_dir, "_".join(map(str, self.masks.squeeze().tolist())))
        # for v in self.missing_view:
        os.makedirs(self.img_save_path_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.model_path_dir, "checkpoint/0/checkpoint_final.pth")

        self.model = VanillaMVVAE(dataset='MMNIST', num_views=self.num_views, num_classes=self.num_classes, 
                                    latent_dim=self.latent_dim, class_dim=self.class_dim,
                                    device=self.device, arch=None, image_shape=(3,28,28), tansformation=[256, 1024, 256])
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['network_weights'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def setup_dataset(self):

        self.dataset_test = generate_dataset(self.data_path, self.num_views, self.missing_rate, dataset='MMNIST', subset='test')

    def setup_classifiers(self):
        cls_path_dir = '/path/to/MVP/pretrained_classifier/trained_clfs_polyMNIST'
        pretrained_classifier_paths = [cls_path_dir + '/pretrained_img_to_digit_clf_m' + str(i) for i in range(self.num_views)]
        self.clfs = {"m%d" % m: None for m in range(self.num_views)}
        for m, fp in enumerate(pretrained_classifier_paths):
            model_clf = ClfImgCMNIST()
            model_clf.load_state_dict(torch.load(fp))
            model_clf = model_clf.to(self.device)
            self.clfs["m%d" % m] = model_clf

    def log_results(self, string):
        print(string)
        with open(self.log_file, 'a') as f:
            f.write(string+'\n')


    def inference(self, image_dict):
        
        image_dict = {k: v.to(self.device).unsqueeze(0) for k, v in image_dict.items()}
        # Two types of latent variables, use matrix for later permutation
        mus = torch.zeros(1, self.num_views, self.num_views, self.latent_dim).to(self.device)
        logvars = torch.zeros(1, self.num_views, self.num_views, self.latent_dim).to(self.device)
        # Encode and transform
        for v in range(self.num_views):
            mu, logvar = self.model.encoders["encoder%d" % v](image_dict["m%d" % v])
            mus[:,v,v,:] = mu
            logvars[:,v,v,:] = logvar
            for j in range(self.num_views):
                if j != v:
                    mus[:,v,j,:] = self.model.transforms_mu["transform%d%d" % (v, j)](mu)
                    logvars[:,v,j,:] = self.model.transforms_logvar["transform%d%d" % (v, j)](logvar)

        idx = torch.nonzero(self.masks[0]).tolist()
        idx = [i[0] for i in idx]
        recon_dict = {}

        mu_combined, logvar_combined = self.model.combination(mus[:, torch.tensor(idx), :, :self.class_dim], logvars[:, torch.tensor(idx), :, :self.class_dim])
        # mu_combined, logvar_combined = mu_combined.mean(dim=1), logvar_combined.mean(dim=1)
        mu_combined, logvar_combined = self.model.combination(mu_combined, logvar_combined)
        for v in range(self.num_views):  
            if v in idx:
                mu_combined_v, logvar_combined_v = self.model.combination(mus[:, v, :, :self.class_dim], logvars[:, v, :, :self.class_dim])
                c = self.model.reparameterize(mu_combined_v, logvar_combined_v)
                z = self.model.reparameterize(mus[:,v,v,:], logvars[:,v,v,:])
                loc = self.model.decoders["decoder%d" % v](self.model.decode_c(c), z)
                recon_dict["m%d" % v] = loc.detach().cpu()
            else:
        
                c = self.model.reparameterize(mu_combined, logvar_combined)    
                mu_, logvar_ = self.model.combination(mus[:,torch.tensor(idx),v,:], logvars[:,torch.tensor(idx),v,:])
                z = self.model.reparameterize(mu_, logvar_) 
                loc = self.model.decoders["decoder%d" % v](self.model.decode_c(c), z)
                recon_dict["m%d" % v] = loc.detach().cpu()
            
        return recon_dict, mu_combined



    def evaluate(self):

        # seed_torch(self.seed)

        
        ssim = {"m%d"%v:[] for v in range(self.num_views)}
        acc = {"m%d"%v:[] for v in range(self.num_views)}
        self.masks = self.masks.to(self.device).unsqueeze(0)
        labels, latents = [], []

        for dataidx in tqdm(range(len(self.dataset_test))):

            acc_i = []
            image_dict, label, _, _, file_name = self.dataset_test[dataidx]
            recon_dict, latent = self.inference(image_dict)
            labels.append(label)
            latents.append(latent)

            for v in range(self.num_views):  
                image_array = (recon_dict["m%d" % v].numpy()+1)/2
                ssim["m%d" % v].append(compare_ssim(image_array[0], (image_dict["m%d" % v].numpy()+1)/2, channel_axis=0, data_range=1))
                clf_input = (recon_dict["m%d" % v]+1)/2
                outcome = self.clfs["m%d" % v](clf_input.to(self.device))
                pred = np.argmax(outcome.detach().cpu().numpy(), axis=1)
                if pred[0] == label:
                    acc["m%d" % v].append(1)
                    acc_i.append(1)
                else:
                    acc["m%d" % v].append(0)
                    acc_i.append(0)   


        average = torch.cat(latents)
        labels = torch.tensor(labels)
        self.kmeans.fit(average.detach().cpu())
        NMI = metrics.normalized_mutual_info_score(labels.numpy(), self.kmeans.labels_)
        self.log_results('Masks: {} Missing view: {}'.format("_".join(map(str, self.masks.squeeze().tolist())), self.missing_view))
        self.log_results('NMI: {:.4f}'.format(NMI))
        self.log_results('Consistency: {:.2f}'.format(np.mean(np.array(consistency))*100))

        ssim_0, ssim_1, acc_mean = [], [], []
        for v in range(self.num_views):
            if v in self.missing_view:
                ssim_0.append(np.mean(np.array(ssim["m%d"%v])))
            else:
                ssim_1.append(np.mean(np.array(ssim["m%d"%v])))
            acc_mean.append(np.mean(np.array(acc["m%d"%v]))*100)
        
        self.log_results('mean: {:.4f}, recon_error 0:{}'.format(mean(ssim_0), ssim_0))
        self.log_results('mean: {:.4f}, recon_error 1: {}'.format(mean(ssim_1), ssim_1))
        self.log_results('mean: {:.2f}, acc: {}'.format(mean(acc_mean), acc_mean))
        self.log_results('========================================================')
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_seed', type=int, default=42, help='Random seed for data')
    parser.add_argument('--missing_view', type=int, nargs='+', help='Views to be missing')
    parser.add_argument('--missing_rate', type=float, default=0.5, help='Rate of missing data')
    parser.add_argument('--class_dim', type=int, default=48, metavar='L',
                    help='the dimension of the latent space')   # You need to change
    parser.add_argument('--latent_dim', type=int, default=96, metavar='L',
                    help='the dimension of the latent space')   # You need to change
    parser.add_argument('--device', type=str, default='cuda:6', help='Device to use for computation')
    parser.add_argument('--model_path_dir', type=str, default=' ', help='Path to the model directory')
    parser.add_argument('--data_path', type=str, default=' ', help='Path to the data directory')
    args = parser.parse_args()

    args.model_path_dir = 'log_MMNIST_missing_0.5_2024-09-25_10-12-52'
    args.data_path = '/path/to/MVP/data/MMNIST'
    missing_combinations = [[[0], [1], [2], [3], [4]],
                        [[0,1], [0,2], [0,3], [0,4], [1,2], [1,3], [1,4], [2,3], [2,4], [3,4]],
                        [[0,1,2],[0,1,3],[0,1,4],[0,2,3],[0,2,4],[0,3,4],[1,2,3],[1,2,4],[1,3,4],[2,3,4]],
                        [[0,1,2,3],[0,1,2,4],[0,1,3,4],[0,2,3,4],[1,2,3,4]]]

    for tasks in missing_combinations: # same number of missing views
        for missing_view in tasks:
            args.missing_view = missing_view
            evaluator = Evaluator(args)
            evaluator.evaluate()


