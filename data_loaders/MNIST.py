from torch.utils import data
from glob import glob
from PIL import Image
import os
import random

class MNIST(data.Dataset):
    def __init__(self, data_dir, which_number, mode, transform):
        if mode != 'test':
            self.image_dirs = sorted(glob(os.path.join(data_dir, f'MNIST_dataset/train/{which_number}', '*.png')))
            num_train = int(len(self.image_dirs) * 0.9)
            if mode == 'train':
                self.image_dirs = self.image_dirs[:num_train]
            elif mode == 'val':
                self.image_dirs = self.image_dirs[num_train:]
        else:
            self.image_dirs = sorted(glob(os.path.join(data_dir, f'MNIST_dataset/test/{which_number}', '*.png')))
        self.transform = transform

    def __getitem__(self, idx):
        img_dir = self.image_dirs[idx]
        img = Image.open(img_dir)
        return self.transform(img)

    def __len__(self):
        return len(self.image_dirs)

    def shuffle_dataset(self):
        random.shuffle(self.image_dirs)


