import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/path/to/MVP')
from model.vanilla_mvVAE_shape import VanillaMVVAE
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
from pretrained_classifier.ConvNetworkImgClfShape import ClfImg as ClfImgShape
import argparse
from statistics import mean

class Evaluator:
    def __init__(self, args):

        self.seed = args.data_seed
        self.model_path_dir = os.path.join('/path/to/MVP/results', args.model_path_dir)
        self.data_path = args.data_path
        self.missing_rate = args.missing_rate
        self.missing_view = args.missing_view
        self.num_views = 5
        self.num_classes = 5
        self.latent_dim = args.latent_dim
        self.class_dim = args.class_dim
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.log_file = os.path.join(self.model_path_dir, 'evaluate_log.txt')
        
        # random.seed(args.data_seed)
        # np.random.seed(args.data_seed)
        
        self.masks = torch.tensor([0 if i in args.missing_view else 1 for i in range(5)])
        
        self.setup_dataset()
        self.setup_classifiers()
        self.setup_model()
        self.kmeans = KMeans(n_clusters = 5, random_state=42, n_init='auto')

    def setup_model(self):
    
        self.img_save_path_dir = os.path.join(self.model_path_dir, "_".join(map(str, self.masks.squeeze().tolist())))
        os.makedirs(self.img_save_path_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.model_path_dir, "checkpoint/0/checkpoint_final.pth")
        self.model = VanillaMVVAE(dataset='MVShapeNet', num_views=self.num_views, num_classes=self.num_classes, 
                                    latent_dim=self.latent_dim, class_dim=self.class_dim,
                                    device=self.device, arch=None, image_shape=(3,64,64), tansformation=[256, 512, 256])
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['network_weights'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def setup_dataset(self):

        self.dataset_test = generate_dataset(self.data_path, self.num_views, self.missing_rate, dataset='MVShapeNet', subset='test')

    def setup_classifiers(self):
        cls_path_dir = '/path/to/MVP/pretrained_classifier/trained_clfs_MVShapeNet/pretrained_img_to_cls_clf.pth'
        view_path_dir = '/path/to/MVP/pretrained_classifier/trained_clfs_MVShapeNet/pretrained_img_to_view_clf.pth'
        model_clf = ClfImgShape()
        model_clf.load_state_dict(torch.load(cls_path_dir)['network_weights'])
        self.classifier_cls = model_clf.to(self.device)
        model_view = ClfImgShape()
        model_view.load_state_dict(torch.load(view_path_dir)['network_weights'])
        self.classifier_view = model_view.to(self.device)

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
        mu_0, logvar_0 = self.model.combination(mus[:,torch.tensor(idx),0,:], logvars[:,torch.tensor(idx),0,:])
        mu_1, logvar_1 = self.model.combination(mus[:,torch.tensor(idx),1,:], logvars[:,torch.tensor(idx),1,:])
        
        for k in range(5):
            a,b = k/4, (1-k/4)
            mu_, logvar_ = b*mu_0+a*mu_1, b*logvar_0+a*logvar_1
            c = self.model.reparameterize(mu_combined, logvar_combined)
            z = self.model.reparameterize(mu_, logvar_)
            loc = self.model.decoders["decoder0"](c, z)
            recon_dict["e%d" % k] = loc.detach().cpu()


        return recon_dict, mu_combined


    def evaluate(self):

        ssim = {"m%d"%v:[] for v in range(self.num_views)}
        acc_cls = {"m%d"%v:[] for v in range(self.num_views)}
        acc_view = {"m%d"%v:[] for v in range(self.num_views)}
        self.masks = self.masks.to(self.device).unsqueeze(0)
        labels, latents = [], []

        for dataidx in tqdm(range(len(self.dataset_test))):

            fig, axes = plt.subplots(1, 5, figsize=(1 * 4, 5 * 4))
            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            image_dict, label, _, _, file_name = self.dataset_test[dataidx]
            recon_dict, latent = self.inference(image_dict)
            labels.append(label)
            latents.append(latent)

            ssim_i = []
            for v in range(self.num_views):  
                image_array = (recon_dict["m%d" % v].numpy()+1)/2
                ssim_idx = compare_ssim(image_array[0], (image_dict["m%d" % v].numpy()+1)/2, channel_axis=0, data_range=1)
                ssim["m%d" % v].append(ssim_idx)

                recon_v = recon_dict["m%d" % v].to(self.device)
                outcome = self.classifier_cls((recon_v+1)/2)
                pred = np.argmax(outcome.detach().cpu().numpy(), axis=1)
                if pred[0] == label:
                    acc_cls["m%d" % v].append(1)
                    # acc_cls_i.append(1)
                else:
                    acc_cls["m%d" % v].append(0)
                    # acc_cls_i.append(0)
                outcome = self.classifier_view((recon_v+1)/2)
                pred = np.argmax(outcome.detach().cpu().numpy(), axis=1)
                if pred[0] == v:
                    acc_view["m%d" % v].append(1)
                    # acc_view_i.append(1)
                else:
                    acc_view["m%d" % v].append(0)
                    # acc_view_i.append(0)

                ssim_i.append(ssim_idx)
            

        average = torch.cat(latents)
        labels = torch.tensor(labels)
        self.kmeans.fit(average.detach().cpu())
        NMI = metrics.normalized_mutual_info_score(labels.numpy(), self.kmeans.labels_)
        self.log_results('Masks: {} Missing view: {}'.format("_".join(map(str, self.masks.squeeze().tolist())), self.missing_view))
        self.log_results('NMI: {:.4f}'.format(NMI))

        ssim_0, ssim_1, acc_cls_mean, acc_view_mean = [], [], [], []
        for v in range(self.num_views):
            if v in self.missing_view:
                ssim_0.append(np.mean(np.array(ssim["m%d"%v])))
            else:
                ssim_1.append(np.mean(np.array(ssim["m%d"%v])))
            acc_cls_mean.append(np.mean(np.array(acc_cls["m%d"%v])))
            acc_view_mean.append(np.mean(np.array(acc_view["m%d"%v])))
        
        self.log_results('mean: {:.4f}, recon_error 0:{}'.format(mean(ssim_0), ssim_0))
        self.log_results('mean: {:.4f}, recon_error 1: {}'.format(mean(ssim_1), ssim_1))
        self.log_results('mean: {:.4f}, acc_cls: {}'.format(mean(acc_cls_mean), acc_cls_mean))
        self.log_results('mean: {:.4f}, acc_view: {}'.format(mean(acc_view_mean), acc_view_mean))
        self.log_results('========================================================')
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_seed', type=int, default=42, help='Random seed for data')
    parser.add_argument('--missing_view', type=int, nargs='+', help='Views to be missing')
    parser.add_argument('--missing_rate', type=float, default=0.5, help='Rate of missing data')
    parser.add_argument('--class_dim', type=int, default=128, metavar='L',
                    help='the dimension of the latent space')   # You need to change
    parser.add_argument('--latent_dim', type=int, default=256, metavar='L',
                    help='the dimension of the latent space')   # You need to change
    parser.add_argument('--device', type=str, default='cuda:3', help='Device to use for computation')
    parser.add_argument('--model_path_dir', type=str, default=' ', help='Path to the model directory')
    parser.add_argument('--data_path', type=str, default=' ', help='Path to the data directory')
    args = parser.parse_args()

    args.model_path_dir = 'log_MVShapeNet_missing_0.5_2024-09-12_13-49-58'
    args.data_path = '/path/to/MVP/data/MVShapeNet'
    missing_combinations = [[[0], [1], [2], [3], [4]],
                        [[0,1], [0,2], [0,3], [0,4], [1,2], [1,3], [1,4], [2,3], [2,4], [3,4]],
                        [[0,1,2],[0,1,3],[0,1,4],[0,2,3],[0,2,4],[0,3,4],[1,2,3],[1,2,4],[1,3,4],[2,3,4]],
                        [[0,1,2,3],[0,1,2,4],[0,1,3,4],[0,2,3,4],[1,2,3,4]]]

    for tasks in missing_combinations: # same number of missing views
        for missing_view in tasks:
            args.missing_view = missing_view
            evaluator = Evaluator(args)
            evaluator.evaluate()


