import numpy as np
import torchvision.transforms as transforms



class TwoAugmentations:

    augment1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    augment2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(0.2),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        transforms.RandomSolarize(0.5, p=0.2),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
        
    def __call__(self, x):
        return self.augment1(x), self.augment2(x)
        


class OneAugmentation:

    augment = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
        
    def __call__(self, x):
        return self.augment(x)