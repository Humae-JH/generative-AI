from model import *
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
from dataLoader import *

batch_size = 64 # batch_size 지정
epoch = 50
num_workers = 0
learning_rate = 0.0001
image_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(image_size),
    transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
])


train_loader = CIFAR10(transform, batch_size, num_workers)

data_mean, data_std = getMeanAndStd(train_loader)
#VAE = model.VAE(device, learning_rate)

#VAE.Train(epoch, train_loader)
#VAE.valid(test_loader)



def Gan_Train():

    G_model = DCGAN(device, learning_rate, batch_size)

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

    G_model.Train(epoch, train_loader)

    G_model.saveState(G_model.discriminator, "DCGAN_Discriminator.pth")
    G_model.saveState(G_model.generator, "DCGAN_Generator.pth")

    return G_model

def VAE_train():
    G_model = VAE(device, learning_rate)
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

    G_model.Train(epoch, train_loader)

    G_model.saveState(G_model.encoder, "VAE_Encoder.pth")
    G_model.saveState(G_model.decoder, "VAE_Decoder.pth")

    return G_model

#G_model = VAE_train()
#Unet = model.UNet(device, learning_rate)
#Unet.to(device)
#Unet.Train(epoch, train_loader)
G_model = DiffusionModel(device, learning_rate)
G_model.to(device)

modelPath = "Diff_model.pth"
if os.path.isfile(modelPath):
    tmp = input(f"{modelPath} has been found would you want to load ? [y / n]...>")
    if tmp == 'y':
        pass
        #G_model.loadState(G_model.network, modelPath)
        #G_model.loadState(G_model.ema_network, modelPath)
    else:
        G_model.Train(epoch, dataloader=train_loader)
        G_model.saveState(G_model.ema_network, "Diff_model.pth")
else:
    G_model.Train(epoch, dataloader=train_loader)
    G_model.saveState(G_model.ema_network, "Diff_model.pth")






#G_model.cpu()
#G_model.setDevice("cpu")

#model = DiffusionModel('cpu',lr=learning_rate)
#model.loadState(model.network, modelPath)
#model.loadState(model.ema_network, modelPath)

#model.eval()
G_model.eval()
loop = 0
while True:
    command = input("quit : q // else : generate image ...>")
    if command == 'q':
        break

    #noise = torch.randn(1, 100)
    noise = torch.nn.functional.normalize(torch.randn([1, 3, image_size, image_size]), dim=1).to(device)
    #noise = torch.normaltorch.rand((1,3,image_size,image_size)).cpu()
    image = G_model.generate(noise, data_mean, data_std)
    image = image.detach().cpu()
    G_model.saveImage(image, "result", f"{loop}th generated image.jpg")
    #G_model.showImage(image)
    loop += 1

