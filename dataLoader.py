from torchvision import datasets, transforms
import torch

def getMeanAndStd(trainloader):
    mean = 0.0
    std = 0.0
    num_batches = len(trainloader)

    # Iterate through the dataset
    for images, _ in trainloader:
        # Flatten the images to get each pixel value in a batch
        images = images.view(images.size(0), images.size(1), -1)
        # Calculate mean and std
        mean += images.mean(2).sum(0) / images.shape[0]
        std += images.std(2).sum(0) / images.shape[0]

    mean = mean / len(trainloader)
    std = std / len(trainloader)

    return mean, std

def CelebA(transforms, batch_size, num_workers):
    dataset = datasets.CelebA(root="./data/celeba",
                              split="train",
                              download=True,
                              transform=transforms)

    # Create the dataloader
    celebDataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
    return celebDataloader


def CIFAR10(transforms,batch_size, num_workers):
    train_dataset = datasets.CIFAR10(root='./data/', train=True, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    """train_data = datasets.FashionMNIST(root='data',
                                       train=True,        # 학습용 데이터셋 설정(True)
                                       download=True,
                                       transform=transform
                                      )

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)"""

    test_data = datasets.CIFAR10(root='./data/', train=False, download=True, transform=transforms)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)

    return train_loader

def MNist(transforms, batch_size):
    train_data = datasets.MNIST(root="./data/", train=True, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)


    return train_loader
