import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Data location
datasetRoot = '/home/selfdriving/datasets/office/domain_adaptation_images/'

# Names
datasetNames = ['amazon', 'dslr', 'webcam']

# Define train and validation phases
phases = ['amazon', 'dslr']

gpu_id = 0

# Training hyper-parameters
ratios = {phases[0]: 0.5, phases[1]: 0.5}
batch_size = 16
init_lr = 0.0003
weight_decay = 0.0005
momentum = 0.9
gamma = 0.001
power = 0.75
maxIter = 60000

# Domain adaptation hyper-parameters
alpha = 10
beta = 1
tau = 0.5

# Data handling
# Rescale and normalize with mean/std
data_transform = {x: transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
    ]) for x in datasetNames}

# Datasets
dsets = {
    x: ImageFolder(os.path.join(os.path.join(datasetRoot, x), 'images'),
                                 data_transform[x])
    for x in datasetNames}

# Dataset sizes
dset_sizes = {x: len(dsets[x]) for x in datasetNames}

# Loaders
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                               shuffle=True, num_workers=4)
                for x in datasetNames}

# Save mode
save_best = 'Training'
