# datasets/cifar10_64x64.py - CIFAR-10 с апскейлом до 64x64 и текстовыми метками
from __future__ import annotations

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CIFAR-10 классы для текстовых промптов
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CIFAR10_CLASS_DESCRIPTIONS = {
    'airplane': 'an airplane flying in the sky',
    'automobile': 'a car or automobile on the road',
    'bird': 'a bird perched or flying',
    'cat': 'a cat sitting or standing',
    'deer': 'a deer in nature',
    'dog': 'a dog sitting or standing',
    'frog': 'a frog on a surface',
    'horse': 'a horse standing or running',
    'ship': 'a ship on water',
    'truck': 'a truck on the road'
}


class CIFAR10_64x64:
    """CIFAR-10 датасет с апскейлом до 64x64 и текстовыми метками"""
    def __init__(self, root='./data', train=True, download=True, use_text=True):
        self.use_text = use_text
        tfm = transforms.Compose([
            transforms.Resize((64, 64)),  # Апскейл до 64x64
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  # [-1,1]
        ])
        self.ds = datasets.CIFAR10(root=root, train=train, download=download, transform=tfm)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, label = self.ds[i]
        if self.use_text:
            class_name = CIFAR10_CLASSES[label]
            text_prompt = CIFAR10_CLASS_DESCRIPTIONS[class_name]
            return img, label, text_prompt, class_name
        else:
            return img, label


def make_loader(batch_size=64, num_workers=4, train=True, use_text=True):
    """Создает DataLoader для CIFAR-10 64x64 с текстовыми метками"""
    ds = CIFAR10_64x64(train=train, use_text=use_text)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
