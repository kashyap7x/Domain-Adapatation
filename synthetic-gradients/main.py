import os
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm
import office_model
import utils


if __name__ == '__main__':
    datasetRoot = 'C:/torch/data/Office31/'
    datasetNames = ['amazon', 'dslr', 'webcam']

    # Define train, validation1 and validation2 phases
    # phases = ['amazon', 'dslr', 'webcam']
    phases = ['dslr', 'amazon', 'webcam']
    # phases = ['webcam', 'amazon', 'dslr']

    gpu_id = 0

    # Training hyper-parameters
    ratios = {phases[0]: 0.8, phases[1]: 0.1, phases[2]: 0.1}
    batch_size = 16
    init_lr = 0.0003
    synth_lr = 0.01
    weight_decay = 0.0005
    momentum = 0.9
    gamma = 0.001
    power = 0.75
    maxIter = 30000

    # Data handling
    # Rescale and normalize with mean/std
    data_transform = {x: transforms.Compose([
            transforms.Scale([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
        ]) for x in datasetNames}
    dsets = {
        x: ImageFolder(os.path.join(os.path.join(datasetRoot, x), 'images'),
                                     data_transform[x])
        for x in datasetNames}
    # Loaders
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=4)
                    for x in datasetNames}
    dset_sizes = {x: len(dsets[x]) for x in datasetNames}

    # Initialize model and trainer
    model = office_model.office_model(init_lr, synth_lr, momentum, weight_decay)
    trainer = office_model.synthetic_trainer(model, phases)

    # Train
    model_best = trainer.train_model(ratios, batch_size, dset_sizes, dset_loaders, utils.inv_lr_scheduler, init_lr, gamma, power, gpu_id=gpu_id, save_best='Training', maxIter=maxIter)
