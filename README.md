# [ICLR 2025] MVP: Multi-View Permutation of Variational Auto-Encoders (MVP)

Welcome to the official implementation of **MVP** (Multi-View Permutation of Variational Auto-Encoders), designed for tackling incomplete multi-view data in representation learning tasks.

## 🚀 Overview

Multi-view learning often hits roadblocks due to missing data in one or more views. MVP steps in with a creative cyclic permutation strategy, paired with a variational inference framework, to establish inter-view correspondences. This enables **sufficient and consistent** representation learning, even with significant data gaps. We've tested MVP on benchmark datasets like PolyMNIST and MVShapeNet, and it delivers strong performance.

---

## 🗂 Code Structure

The project is neatly organized into the following directories:

```
MVP/
│
├── data/                      # Data loaders and dataset preprocessing
├── dataprovider/              # Functions for handling multi-view data and missing patterns
├── evaluate/                  # Evaluation metrics and visualization utilities
├── model/                     # Model architectures (e.g., encoders, decoders, and correspondence networks)
├── pretrained_classifier/     # Pre-trained classifiers for downstream tasks
├── results/                   # Outputs including logs, checkpoints, and visualizations
├── training/                  # Training scripts and loss function implementations
├── main.py                    # Entry point for training the model
├── main_img.py                # Entry point for image-based datasets like PolyMNIST and MVShapeNet
├── utils.py                   # Helper functions for logging, visualization, and utility operations
└── README.md                  # Project documentation
```

---

## 🎯 Training Command

To train the model on the **Handwritten** dataset with a 50% missing rate, run the following command:

```bash
CUDA_VISIBLE_DEVICES=1 python /path/to/MVP/main.py --dataset 'Handwritten' --missing_rate 0.5
```

For the **PolyMNIST** dataset, place it in the `MMNIST` folder and run:

```bash
CUDA_VISIBLE_DEVICES=1 python /path/to/MVP/main_img.py --dataset 'MMNIST' --missing_rate 0.5
```

---

## 🌟 Features

### Current Implementation
- Handling missing views with cyclic permutations.
- Variational inference framework with inter-view correspondences.
- Full implementation of MVP on PolyMNIST and other benchmark datasets.

### 📊 Visualization
- Loss evolution curves illustrating training dynamics.
- Latent space visualizations showing inter-view correspondence.

---

## 🛠 Future Work (TODO List)

- [ ] Release the preprocessed **MVShapeNet** dataset along with detailed usage instructions.
- [ ] Extend the implementation to additional datasets with diverse view characteristics.
- [ ] Optimize the model for improved training efficiency and scalability.
- [ ] Provide a detailed analysis of missing patterns and their impact on model performance.

---

## 📜 Citation

If you find this repository helpful, please consider citing:

```
@inproceedings{
gao2025deep,
title={Deep Incomplete Multi-view Learning via Cyclic Permutation of {VAE}s},
author={Xin Gao and Jian Pu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=s4MwstmB8o}
}
```

---

Feel free to share any feedback or raise issues for additional support. Thank you for your interest in our work!

--- 
