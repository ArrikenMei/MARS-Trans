import torch
import torchvision
import torchvision.transforms as transforms
import TestModel.ViT.config as config
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from PIL import Image


def data_process(height, weight, batch_size):
    train_transforms = transforms.Compose([
        # torchvision.transforms.Resize(size, interpolation=2)
        # size（sequence 或int） -所需的输出大小。如果size是类似（h，w）的序列，则输出大小将与此匹配。如果size是int，则图像的较小边缘将与此数字匹配。即，如果高度>宽度，则图像将重新缩放为（尺寸*高度/宽度，尺寸）
        # interpolation（int，optional） - 所需的插值。默认是 PIL.Image.BILINEAR
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
    # ImageFolder是一个通用的数据加载器，它要求我们以(root/dog/xxx.png)这种格式来组织数据集的训练、验证或者测试图片。
    train_data = torchvision.datasets.ImageFolder(root='E:\\DLmaster\\MyCNN\\datasets\\cub200\\train',
                                                  transform=train_transforms)
    test_data = torchvision.datasets.ImageFolder(root='E:\\DLmaster\\MyCNN\\datasets\\cub200\\test',
                                                 transform=test_transforms)

    # 把训练数据分成多个小组 ，此函数 每次抛出一组数据 。直至把所有的数据都抛出。就是做一个数据的初始化。
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
