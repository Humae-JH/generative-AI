import model
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

batch_size = 128 # batch_size 지정
epoch = 4
num_workers = 0
learning_rate = 0.0002
image_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(image_size),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

train_dataset = datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle=True)

"""train_data = datasets.FashionMNIST(root='data',
                                   train=True,        # 학습용 데이터셋 설정(True)
                                   download=True,
                                   transform=transform
                                  )

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers)"""

test_data = datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers)

#VAE = model.VAE(device, learning_rate)

#VAE.Train(epoch, train_loader)
#VAE.valid(test_loader)

DCGan = model.DCGAN(device, learning_rate, batch_size)
DCGan.Train(epoch, train_loader)

loop = 0
while True:
    command = input("quit : q // else : generate image ...>")
    if command == 'q':
        break

    noise = torch.randn(1, 100)
    image = DCGan.generate(noise)
    DCGan.saveImage(image, "result", f"{loop}th generated image.jpg")
    DCGan.showImage(image)
    loop += 1

