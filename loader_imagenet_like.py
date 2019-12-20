
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data
from PIL import Image
import os
from glob import glob
import os.path
import errno
import numpy as np
import sys
if sys.version_info[0] == 2:
    import pickle as pickle
else:
    import pickle


class ImagenetLike(data.Dataset):
    split_list = ['label', 'unlabel', 'valid', 'test']

    def __init__(self, root, split='train',
                 transform=None, target_transform=None,
                 download=False, boundary=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        assert(boundary < 10)

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        sub = ''
        if self.split == 'label':
            sub = 'train'
        elif self.split == 'unlabel':
            sub = 'validation'
        elif self.split == 'valid' or self.split == 'test':
            sub = 'test'

        self.files = np.array(
            sorted(glob(os.path.join(self.root, sub, '**', '*.jpg'))))

        labels = [x.split(os.sep)[-2] for x in self.files]
        classes = list(np.unique(labels))
        self.labels = np.array([classes.index(x) for x in labels])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target, img1 = Image.open(
            self.files[index]).convert(mode='RGB'), self.labels[index], Image.open(self.files[index]).convert(mode='RGB')

        if self.transform is not None:
            img = self.transform(img)
            img1 = self.transform(img1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, img1

    def __len__(self):
        print(len(self.files))
        return len(self.files)
