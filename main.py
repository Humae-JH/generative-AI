import model
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os

batch_size = 128 # batch_size 지정
epoch = 10
num_workers = 0
learning_rate = 0.0002
image_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

dataset = datasets.CelebA(root="./data/celeba",
                      split="train",
                      download=True,
                      transform=transforms.Compose([
                          transforms.Resize(image_size),
                          transforms.CenterCrop(image_size),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ]))

# Create the dataloader
celebDataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers)

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



def Gan_Train():

    G_model = model.DCGAN(device, learning_rate, batch_size)

    D_weight_path = "./DCGAN_Discriminator.pth"
    G_weight_path = "./DCGAN_Generator.pth"

    if os.path.isfile(D_weight_path) :
        tmp = input(f"{D_weight_path} has been found would you want to load ? [y / n]...>")
        if tmp == 'y' :
            G_model.loadState(G_model.discriminator, D_weight_path)
        else:
            pass


    if os.path.isfile(G_weight_path):
        tmp = input(f"{G_weight_path} has been found would you want to load ? [y / n]...>")
        if tmp == 'y' :
            G_model.loadState(G_model.generator, G_weight_path)
        else:
            pass

    G_model.Train(epoch, celebDataloader)

    G_model.saveState(G_model.discriminator, "DCGAN_Discriminator.pth")
    G_model.saveState(G_model.generator, "DCGAN_Generator.pth")

    return G_model

def VAE_train():
    G_model = model.VAE(device, learning_rate)
    E_weight_path = "./VAE_Encoder.pth"
    D_weight_path = "./VAE_Decoder.pth"

    if os.path.isfile(D_weight_path):
        tmp = input(f"{D_weight_path} has been found would you want to load ? [y / n]...>")
        if tmp == 'y':
            G_model.loadState(G_model.discriminator, D_weight_path)
        else:
            pass

    if os.path.isfile(E_weight_path):
        tmp = input(f"{E_weight_path} has been found would you want to load ? [y / n]...>")
        if tmp == 'y':
            G_model.loadState(G_model.generator, E_weight_path)
        else:
            pass

    G_model.Train(epoch, celebDataloader)

    G_model.saveState(G_model.encoder, "VAE_Encoder.pth")
    G_model.saveState(G_model.decoder, "VAE_Decoder.pth")

    return G_model

G_model = VAE_train()

loop = 0
while True:
    command = input("quit : q // else : generate image ...>")
    if command == 'q':
        break
    G_model.to('cpu')
    G_model.eval()
    noise = torch.randn(1, 100)
    image = G_model.generate(noise)
    G_model.saveImage(image, "result", f"{loop}th generated image.jpg")
    G_model.showImage(image)
    loop += 1

