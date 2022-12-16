import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import transforms as T

import sys
sys.path.append('../')

def get_transforms(inputs, subset='train'):
    aug_name = inputs.AUGMENTATION_NAME
    if aug_name == 'imagenet':
        print('Using ImageNet augmentations ...')
        if subset == 'train':
            return T.Compose([
                T.RandomResizedCrop(inputs.img_size),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                # TODO: add PCA noise
                T.ToTensor(),
                T.Normalize(inputs.MEAN, inputs.STDV)
            ])
        else:
            return T.Compose([
                T.Resize((inputs.img_size, inputs.img_size)),
                T.ToTensor(),
                T.Normalize(inputs.MEAN, inputs.STDV)
            ])
    elif aug_name == 'augmentation_bia':
        if subset == 'train':
            return A.Compose([
                A.Resize(height=inputs.img_size, width=inputs.img_size, interpolation=cv2.INTER_CUBIC),
                A.GaussNoise(always_apply=False, p=1.0, var_limit=(0.001)),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=inputs.MEAN, std=inputs.STDV),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(height=inputs.img_size, width=inputs.img_size, interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean=inputs.MEAN, std=inputs.STDV),
                ToTensorV2()
            ])
    elif aug_name == 'auto_trivial':
        if subset == 'train':
            return T.Compose([
                T.Resize((inputs.img_size, inputs.img_size)),
                T.TrivialAugmentWide(),
                T.ToTensor(),
                T.Normalize(inputs.MEAN, inputs.STDV)
            ])
        else:
            return T.Compose([
                T.Resize((inputs.img_size, inputs.img_size)),
                T.ToTensor(),
                T.Normalize(inputs.MEAN, inputs.STDV)
            ])
    elif aug_name == 'small_complexity':
        if subset == 'train':
            return A.Compose([
                A.ShiftScaleRotate(
                    shift_limit=0.03,
                    scale_limit=0.05,
                    rotate_limit=4,
                    interpolation=cv2.INTER_CUBIC,
                    p=0.9
                ),
                A.Resize(height=inputs.img_size, width=inputs.img_size, interpolation=cv2.INTER_CUBIC),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=inputs.MEAN, std=inputs.STDV),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(height=inputs.img_size, width=inputs.img_size, interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean=inputs.MEAN, std=inputs.STDV),
                ToTensorV2()
            ])
    else:
        print('Using only horizontal flip augmentation.')
        if subset == 'train':
            return A.Compose([
                A.Resize(height=inputs.img_size, width=inputs.img_size, interpolation=cv2.INTER_CUBIC),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=inputs.MEAN, std=inputs.STDV),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(height=inputs.img_size, width=inputs.img_size, interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean=inputs.MEAN, std=inputs.STDV),
                ToTensorV2()
            ])
