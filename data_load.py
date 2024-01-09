import torch
import torchvision
import torchvision.transforms as transforms
import TestModel.ViT.config as config
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from PIL import Image


def data_process(height, weight, batch_size):
    train_transforms = transforms.Compose([
        # torchvision.transforms.Resize(size, interpolation=2)
        transforms.Resize(600),
        transforms.RandomCrop((448, 448)),
        transforms.Resize((height, weight)),
        # transforms.RandomResizedCrop((height, weight)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomGaussianBlur,
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_transforms = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop((448, 448)),
        transforms.Resize((height, weight)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_data = torchvision.datasets.ImageFolder(root='E:\\DLmaster\\MyCNN\\datasets\\cub200\\train',
                                                  transform=train_transforms)
    test_data = torchvision.datasets.ImageFolder(root='E:\\DLmaster\\MyCNN\\datasets\\cub200\\test',
                                                 transform=test_transforms)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=True)

    return train_loader, test_loader
