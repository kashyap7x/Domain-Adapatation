import torch
import torchvision
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm


def resnetOutput(model_trained, x):
    x = model_trained.conv1(x)
    x = model_trained.bn1(x)
    x = model_trained.relu(x)
    x = model_trained.maxpool(x)

    x = model_trained.layer1(x)
    x = model_trained.layer2(x)
    x = model_trained.layer3(x)
    x = model_trained.layer4(x)

    x = model_trained.avgpool(x)
    x = x.view(x.size(0), -1)
    return x[0]


if __name__ == '__main__':

    datasetRoot = '/home/selfdriving/datasets/office/domain_adaptation_images/'
    datasetNames = ['amazon', 'dslr','webcam']
    phases = ['webcam', 'amazon', 'dslr']
    gpu_id = 1

    data_transform = {x: transforms.Compose([
            transforms.Scale([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
        ]) for x in datasetNames}

    dsets = {
        x: ImageFolder(os.path.join(os.path.join(datasetRoot, x), 'images'),
                                     data_transform[x])
        for x in datasetNames}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=1,
                                                   shuffle=False, num_workers=4)
                    for x in datasetNames}
    dset_sizes = {x: len(dsets[x]) for x in datasetNames}

    model_trained = torch.load(phases[0]+'.pt')
    model_trained.eval()
    model_trained.cuda(gpu_id)
    
    resnet_features_phase1 = np.zeros((len(dsets[phases[0]]),2048))
    resnet_features_phase2 = np.zeros((len(dsets[phases[1]]),2048))
    resnet_features_phase3 = np.zeros((len(dsets[phases[2]]),2048))
    labels1 = np.zeros(len(dsets[phases[0]]))
    labels2 = np.zeros(len(dsets[phases[1]]))
    labels3 = np.zeros(len(dsets[phases[2]]))

    for phase in phases:
        # Iterate over data.
        for loc, data in enumerate(tqdm(dset_loaders[phase])):
            # get the inputs
            inputs, target = data
            # wrap them in Variable
            if gpu_id >= 0:
                inputs, target = Variable(inputs.cuda(gpu_id)), Variable(target.cuda(gpu_id))
            else:
                inputs, target = Variable(inputs), Variable(target)
            
            if(phase == phases[0]):
                resnet_features_phase1[loc,:] = resnetOutput(model_trained,inputs)
                labels1[loc] = target
            if(phase == phases[1]):
                resnet_features_phase2[loc,:] = resnetOutput(model_trained,inputs)
                labels2[loc] = target
            if(phase == phases[2]):
                resnet_features_phase3[loc,:] = resnetOutput(model_trained,inputs)
                labels3[loc] = target
            
    np.save('features_from_' + phases[0]+'_to_'+phases[0], resnet_features_phase1)
    np.save('features_from_' + phases[0]+'_to_'+phases[1], resnet_features_phase2)
    np.save('features_from_' + phases[0]+'_to_'+phases[2], resnet_features_phase3)
    np.save('labels_' + phases[0], labels1)
    np.save('labels_' + phases[1], labels2)
    np.save('labels_' + phases[2], labels3)
  