import os
import cv2

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import sys
sys.path.append('..')
from preprocessing.utils import crop_image, remove_border


class RadiographSexDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        fold_nums: list,
        transforms,
        albumentations_package: bool=True,
        crop_side: str=None,
        border: int=0
    ):
        super().__init__()

        self.root_dir = root_dir
        self.fold_nums = fold_nums
        self.transforms = transforms
        self.albumentations = albumentations_package
        self.crop_side = crop_side

        # labels
        self.filepaths = []
        for fold_num in self.fold_nums:
            foldpath = os.path.join(self.root_dir, f'fold-{fold_num:02d}')
            for filename in os.listdir(foldpath):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    self.filepaths.append(os.path.join(foldpath, filename))

        # this maybe useful later for reproducibility
        self.filepaths.sort()

    def __len__(self) -> int:
        return len(self.filepaths)

    def _getitem_albumentations(self, image):
        # Read image with OpenCV2 and convert it from BGR (OpenCV2) to RGB (most common format)
        image = np.array(image)

        # apply transformation with albumentations package
        if self.transforms is not None:
            img_tensor = self.transforms(image=image)["image"]

        return img_tensor

    def _getitem_torchvision(self, image):

        if self.transforms is not None:
            img_tensor = self.transforms(image)

        return img_tensor

    def __getitem__(self, index: int):
        # image and label
        filepath = self.filepaths[index]

        # get label
        filename = filepath.split('/')[-1]
        gender = filename.split('-')[7]

        assert gender in ['F', 'M']
        if gender == 'F':
            label = 0
        else:
            label = 1

        label_tensor = torch.tensor(label, dtype=torch.int64)

        image = Image.open(filepath)
        image = remove_border(image)
        if self.crop_side:
            image = crop_image(image, self.crop_side)

        # apply transforms
        if self.albumentations:
            img_tensor = self._getitem_albumentations(image)
        else:
            img_tensor = self._getitem_torchvision(image)

        return img_tensor, label_tensor


class FullRadiographSexDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        fold_nums: list,
        transforms,
        fold_txt_dir: str='splits',
        albumentations_package: bool=True
    ):
        super().__init__()

        self.root_dir = root_dir
        self.fold_nums = fold_nums
        self.transforms = transforms
        self.albumentations = albumentations_package

        # labels
        self.filepaths = []
        for i in fold_nums:
            filepath = os.path.join(root_dir, fold_txt_dir, f'{i:02d}.txt')
            with open(filepath) as txt_file:
                for line in txt_file:
                    img_relpath = line.strip()
                    filename = img_relpath.split('/')[-1]
                    sex = filename.split('-')[10]
                    if sex not in ['M', 'F']:
                        continue
                    self.filepaths.append(os.path.join(root_dir, img_relpath))

        # this maybe useful later for reproducibility
        self.filepaths.sort()

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int):
        # image and label
        filepath = self.filepaths[index]
        filename = filepath.split('/')[-1]

        # get the labels
        sex = filename.split('-')[10]
        # age = filename.split('-')[-2][1:]
        # months = filename.split('-')[-1][1:3]

        assert sex in ['F', 'M']
        if sex == 'F':
            label = 0
        else:
            label = 1

        label_tensor = torch.tensor(label, dtype=torch.int64)

        image = Image.open(filepath)
        image = image.convert('RGB')

        # apply transforms
        if self.albumentations:
            image = np.array(image)
            img_tensor = self.transforms(image=image)["image"]
        else:
            raise Exception('Not implemented yet.')
            
        return img_tensor, label_tensor