import os
import random
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
from scipy.misc import imread, imresize
from tqdm import tqdm


# dataset list generation functions
def make_GTA(root):
    image_root = os.path.join(root, 'playing')
    mask_root = os.path.join(root, 'playing_annotations')
    items = []
    imgdirs = os.listdir(image_root)
    for filename in imgdirs:
        imgpath = os.path.join(image_root, filename)
        maskpath = os.path.join(mask_root, filename)
        items.append((imgpath, maskpath))
    return items


def make_CityScapes(mode, root):
    img_dir_name = 'leftImg8bit_trainvaltest'
    mask_path = os.path.join(root, 'gtFine_trainvaltest', 'gtFine', mode)
    mask_postfix = '_gtFine_labelIds.png'
    img_path = os.path.join(root, img_dir_name, 'leftImg8bit', mode)
    assert os.listdir(img_path) == os.listdir(mask_path)
    items = []
    categories = os.listdir(img_path)
    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + mask_postfix))
            items.append(item)
    return items


def make_BDD(mode, root):
    mask_path = os.path.join(root, 'seg_maps', 'labels', mode)
    mask_postfix = '_train_id.png'
    img_path = os.path.join(root, 'images', '10k', mode)
    img_postfix = '.jpg'
    items = []
    filenames = [name.split(img_postfix)[0] for name in os.listdir(img_path)]
    for it in filenames:
        item = (os.path.join(img_path, it + img_postfix), os.path.join(mask_path, it + mask_postfix))
        items.append(item)
    return items


class GTA(torchdata.Dataset):
    def __init__(self, root, cropSize=720, ignore_label=-1, max_sample=-1, is_train=1):
        self.list_sample = make_GTA(root)
        self.is_train = is_train
        self.cropSize = cropSize
        self.ignore_label = ignore_label
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.id_to_trainid = {34: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

        if self.is_train:
            random.shuffle(self.list_sample)
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def _scale_and_crop(self, img, seg, cropSize, is_train):
        h_s, w_s = 720, 1312
        img_scale = imresize(img, (h_s, w_s), interp='bilinear')
        seg = (seg + 1).astype(np.uint8)
        seg_scale = imresize(seg, (h_s, w_s), interp='nearest')
        seg_scale = seg_scale.astype(np.int) - 1

        if is_train:
            # random crop
            x1 = random.randint(0, w_s - cropSize)
            y1 = random.randint(0, h_s - cropSize)
            img_crop = img_scale[y1: y1 + cropSize, x1: x1 + cropSize, :]
            seg_crop = seg_scale[y1: y1 + cropSize, x1: x1 + cropSize]
        else:
            # no crop
            img_crop = img_scale
            seg_crop = seg_scale

        return img_crop, seg_crop

    def _flip(self, img, seg):
        img_flip = img[:, ::-1, :].copy()
        seg_flip = seg[:, ::-1].copy()
        return img_flip, seg_flip

    def __getitem__(self, index):
        img_path, seg_path = self.list_sample[index]
        
        img = imread(img_path, mode='RGB')
        seg = imread(seg_path, mode='P')
        
        seg_copy = seg.copy().astype(np.int)
        for k, v in self.id_to_trainid.items():
            seg_copy[seg == k] = v
        seg = seg_copy
        
        # random scale, crop, flip
        img, seg = self._scale_and_crop(img, seg,
                                        self.cropSize, self.is_train)
        if self.is_train and random.choice([-1, 1]) > 0:
            img, seg = self._flip(img, seg)

        # image to float
        img = img.astype(np.float32) / 255.
        img = img.transpose((2, 0, 1))
        
        # to torch tensor
        image = torch.from_numpy(img)
        segmentation = torch.from_numpy(seg)

        # normalize
        image = self.img_transform(image)

        return image, segmentation.long(), img_path

    def __len__(self):
        return len(self.list_sample)


class CityScapes(torchdata.Dataset):
    def __init__(self, mode, root, cropSize=720, ignore_label=-1, max_sample=-1, is_train=1):
        self.list_sample = make_CityScapes(mode, root)
        self.mode = mode
        self.cropSize = cropSize
        self.is_train = is_train
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        if self.is_train:
            random.shuffle(self.list_sample)
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def _scale_and_crop(self, img, seg, cropSize, is_train):
        h_s, w_s = 720, 1440
        img_scale = imresize(img, (h_s, w_s), interp='bilinear')
        seg = (seg + 1).astype(np.uint8)
        seg_scale = imresize(seg, (h_s, w_s), interp='nearest')
        seg_scale = seg_scale.astype(np.int) - 1

        if is_train:
            # random crop
            x1 = random.randint(0, w_s - cropSize)
            y1 = random.randint(0, h_s - cropSize)
            img_crop = img_scale[y1: y1 + cropSize, x1: x1 + cropSize, :]
            seg_crop = seg_scale[y1: y1 + cropSize, x1: x1 + cropSize]
        else:
            # no crop
            img_crop = img_scale
            seg_crop = seg_scale

        return img_crop, seg_crop

    def _flip(self, img, seg):
        img_flip = img[:, ::-1, :].copy()
        seg_flip = seg[:, ::-1].copy()
        return img_flip, seg_flip

    def __getitem__(self, index):
        img_path, seg_path = self.list_sample[index]

        img = imread(img_path, mode='RGB')
        seg = imread(seg_path, mode='P')
        
        seg_copy = seg.copy().astype(np.int)
        for k, v in self.id_to_trainid.items():
            seg_copy[seg == k] = v
        seg = seg_copy

        # random scale, crop, flip
        img, seg = self._scale_and_crop(img, seg,
                                        self.cropSize, self.is_train)
        if self.is_train and random.choice([-1, 1]) > 0:
            img, seg = self._flip(img, seg)

        # image to float
        img = img.astype(np.float32) / 255.
        img = img.transpose((2, 0, 1))
        
        # to torch tensor
        image = torch.from_numpy(img)
        segmentation = torch.from_numpy(seg)

        # normalize
        image = self.img_transform(image)

        return image, segmentation.long(), img_path

    def __len__(self):
        return len(self.list_sample)


class BDD(torchdata.Dataset):
    def __init__(self, mode, root, cropSize=720, ignore_label=-1, max_sample=-1, is_train=1):
        self.list_sample = make_BDD(mode, root)
        self.mode = mode
        self.cropSize = cropSize
        self.is_train = is_train
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.id_to_trainid = {255:ignore_label}
        if self.is_train:
            random.shuffle(self.list_sample)
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def _scale_and_crop(self, img, seg, cropSize, is_train):
        h_s, w_s = 720, 1280
        if img.shape[0] == h_s and img.shape[1] == w_s:
            img_scale = img
            seg_scale = seg
        else:
            img_scale = imresize(img, (h_s, w_s), interp='bilinear')
            seg = (seg + 1).astype(np.uint8)
            seg_scale = imresize(seg, (h_s, w_s), interp='nearest')
            seg_scale = seg_scale.astype(np.int) - 1

        if is_train:
            # random crop
            x1 = random.randint(0, w_s - cropSize)
            y1 = random.randint(0, h_s - cropSize)
            img_crop = img_scale[y1: y1 + cropSize, x1: x1 + cropSize, :]
            seg_crop = seg_scale[y1: y1 + cropSize, x1: x1 + cropSize]
        else:
            # no crop
            img_crop = img_scale
            seg_crop = seg_scale

        return img_crop, seg_crop

    def _flip(self, img, seg):
        img_flip = img[:, ::-1, :].copy()
        seg_flip = seg[:, ::-1].copy()
        return img_flip, seg_flip

    def selfcheck(self, list_sample):
        counter = 0
        filtered_list = []
        for i in tqdm(range(len(list_sample))):
            img_path, seg_path = self.list_sample[i]
            img = imread(img_path, mode='RGB')
            seg = imread(seg_path, mode='P')
            if img.shape[0] != seg.shape[0] or img.shape[1] != seg.shape[1]:
                counter += 1
            else:
                filtered_list.append((img_path, seg_path))
        print('{} ignored samples (mismatch in input and label)'.format(counter))

        return filtered_list

    def __getitem__(self, index):
        img_path, seg_path = self.list_sample[index]

        img = imread(img_path, mode='RGB')
        seg = imread(seg_path, mode='P')

        seg_copy = seg.copy().astype(np.int)
        for k, v in self.id_to_trainid.items():
            seg_copy[seg == k] = v
        seg = seg_copy

        # random scale, crop, flip
        img, seg = self._scale_and_crop(img, seg,
                                        self.cropSize, self.is_train)
        if self.is_train and random.choice([-1, 1]) > 0:
            img, seg = self._flip(img, seg)

        # image to float
        img = img.astype(np.float32) / 255.
        img = img.transpose((2, 0, 1))

        # to torch tensor
        image = torch.from_numpy(img)
        segmentation = torch.from_numpy(seg)

        # normalize
        image = self.img_transform(image)

        if image.shape[1] != segmentation.shape[0] or image.shape[2] != segmentation.shape[1]:
            print('test')
        return image, segmentation.long(), img_path

    def __len__(self):
        return len(self.list_sample)