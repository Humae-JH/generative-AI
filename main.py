import model
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

batch_size = 128 # batch_size 지정
epoch = 10
num_workers = 0
learning_rate = 0.0001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(32),
])

train_data = datasets.FashionMNIST(root='data',
                                   train=True,        # 학습용 데이터셋 설정(True)
                                   download=True,
                                   transform=transform
                                  )

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers)

test_data = datasets.FashionMNIST(root='data',
                                  train=False,        # 검증용 데이터셋 설정(False)
                                  download=True,
                                  transform=transform
                                 )

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers)

VAE = model.VAE(device, learning_rate)

VAE.Train(epoch, train_loader)
VAE.valid(test_loader)


