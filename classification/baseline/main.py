import torch
import torch.optim as optim
import torchvision
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
import office_model


def inv_lr_scheduler(optimizer, gamma, power, iter, init_lr=0.001):
    lr = init_lr * (1 + gamma * iter) ** (- power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer, lr


if __name__ == '__main__':

    datasetRoot = '/home/selfdriving/datasets/office/domain_adaptation_images/'
    datasetNames = ['amazon', 'dslr', 'webcam']

    # phases = ['amazon', 'dslr', 'webcam']
    phases = ['dslr', 'amazon', 'webcam']
    # phases = ['webcam', 'amazon', 'dslr']
    batch_size = 16
    init_lr = 0.0003
    weight_decay = 0.0005
    gpu_id = 0
    num_epochs = 40
    gamma = 0.001
    power = 0.75
    maxIter = 30000

    data_transform = {x: transforms.Compose([
            transforms.Scale([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
        ]) for x in datasetNames}

    dsets = {
        x: ImageFolder(os.path.join(os.path.join(datasetRoot, x), 'images'),
                                     data_transform[x])
        for x in datasetNames}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=4)
                    for x in datasetNames}
    dset_sizes = {x: len(dsets[x]) for x in datasetNames}

    model = office_model.office_model()

    optimizer = optim.SGD(model.parameters(), lr=init_lr, weight_decay=weight_decay)

    model_best = model.train_model(model, dset_loaders, dset_sizes, optimizer, inv_lr_scheduler, init_lr, phases, gamma, power, gpu_id=gpu_id, maxIter=maxIter)

    torch.save(model_best.resnet, (phases[0] + '.pt'))

