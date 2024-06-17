from torch import nn
import torch
from abc import *
from torch.optim import *
import torchvision
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, device, lr):
        super().__init__()
        self.device = device
        self.learning_rate = lr
        pass

    @abstractmethod
    def forward(self,x):
        pass

    def Train(self, epoch, dataloader):
        pass

    def valid(self):
        pass

    @abstractmethod
    def generate(self, z):
        pass

    def saveImage(self, images, output_dir, image_name):
        img_grid = vutils.make_grid(images, padding=2, normalize=True)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, image_name)
        vutils.save_image(img_grid, save_path, normalize=True)
        return

    def showImage(self, images):
        img_grid = vutils.make_grid(images, padding=2, normalize=True)
        # Show the images
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.title(f'Generated Images')
        plt.imshow(np.transpose(img_grid, (1, 2, 0)))
        plt.show()

    def showLossGraph(self, losses, loss_name):
        plt.figure(figsize=(10, 5))
        plt.title(f"{ loss_name } Loss During Training")
        plt.plot(losses, label=loss_name)
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

class VAEEncoder(BaseModel):
    def __init__(self, device, lr):
        super().__init__(device, lr)
        self.network = nn.Sequential(
                nn.Conv2d(1, 32, 3, 2, 1), # output : H, W, C = 16, 16, 32)
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(64, 128,3, 2, 1), #4, 4, 128
                nn.ReLU(),
                nn.Flatten(1, -1)
        )

        self.z_mean = nn.Linear(4*4*128, 10)
        self.z_log_var = nn.Linear(4*4*128, 10)

    def klLoss(self):
        return -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
    def forward(self, x):
        out = self.network(x)
        self.mu = self.z_mean(out)
        self.logvar = self.z_log_var(out)
        return self.reparameterization(self.mu, self.logvar), self.mu, self.logvar

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEDecoder(BaseModel):
    def __init__(self, device, lr):
        super().__init__(device, lr)
        self.linear = nn.Linear(10, 2048)

        self.network = nn.Sequential(
            nn.Unflatten(1, torch.Size([128,4,4])),
            nn.ConvTranspose2d(128,64, 5,1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):

        out = self.linear(z)
        out = self.network(out)
        return out


class VAE(BaseModel):
    def __init__(self, device, lr):
        super().__init__(device, lr)

        self.encoder = VAEEncoder(device, lr).to(self.device)
        self.decoder = VAEDecoder(device, lr).to(self.device)

        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)
        #self.KLloss = torch.nn.KLDivLoss()
        self.Reconloss = torch.nn.BCELoss()

    def forward(self, x):
        encoder_out, mu, logvar = self.encoder.forward(x)
        decoder_out = self.decoder.forward(encoder_out)
        return decoder_out

    def Train(self, epoch, dataloader):
        self.train()
        for i in range(0, epoch):
            train_loss = 0
            print(f"[train epoch] : {i} ")
            for  _, (x,y) in enumerate(dataloader):
                self.zero_grad()
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                #y_hat = self.forward(x)
                z, mu, logvar = self.encoder.forward(x)
                y_hat = self.decoder.forward(z)
                klloss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = klloss + self.Reconloss(y_hat, x)
                train_loss += loss.item()
                if _ % 100 == 0:
                    print(f"train_loss : {train_loss / len(dataloader.dataset)}")
                loss.backward()
                self.optimizer.step()




        return


    def valid(self, dataloader):
        self.eval()
        val_loss = 0

        with torch.no_grad():
            for  _, (x,y) in enumerate(dataloader):
                x = x.to(self.device)
                torchvision.utils.save_image(x,f"./original image {_}.jpg")
                output = self.forward(x)
                torchvision.utils.save_image(output, f"./generated image {_}.jpg")

                loss = self.encoder.klLoss() + self.Reconloss(output, x)
                val_loss += loss.item()

        avg_loss = val_loss / len(dataloader.dataset)
        print(f"validation loss : {avg_loss}")


    def generate(self,z):
        image = self.decoder.forward(z)
        return image

class DCGAN(BaseModel):
    def __init__(self, device, lr, batch_size):
        super().__init__(device, lr)
        self.batch_size = batch_size
        self.dropout = 0.0
        """ Pooling 계층을 사용하는것 보다 Stride를 사용하는 것이 자체 pool을 학습해서 더 유리하다! """
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), # 64 -> 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout),
            nn.Conv2d(64, 128, 3, 2, 1), # 32-> 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout),
            nn.Conv2d(128, 256, 3, 2, 1), # 16 -> 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout),
            nn.Conv2d(256, 512, 3, 2, 1), # 8 -> 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0), # 4 -> 1
            nn.Sigmoid(),
            #nn.Conv2d(128, 256, 4, 1,  0), # 4 -> 1
            #nn.BatchNorm2d(256),
            #nn.LeakyReLU(0.2),
            #nn.Conv2d(256, 1, 1, 1, 0),
            #nn.Sigmoid(),
        )

        self.generator = nn.Sequential(
            nn.Linear(100, 1024*4*4),
            nn.Unflatten(1, torch.Size([1024,4,4])),
            nn.ConvTranspose2d(1024,512, 4 ,2, 1), # 4 -> 8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 4 -> 8 ( 계산식 : (input - 1) * stride - 2 * padding + (kernel_size - 1) + outpadding + 1
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.ConvTranspose2d(128, 3, 4, 2, 1), # 16 -> 32
            nn.Tanh(),
            #nn.BatchNorm2d(64),
            #nn.ReLU(),
            #nn.Dropout(self.dropout),
            #nn.ConvTranspose2d(64, 3, 4, 2, 1), ## 32 -> 64
            #nn.Tanh(),
        )
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        self.optimizerG = Adam(self.generator.parameters(), 0.001, betas=(0.5, 0.999))
        self.optimizerD = Adam(self.discriminator.parameters(), 0.0002, betas=(0.5, 0.999))

        self.loss = nn.BCELoss()

    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def Train(self, epoch, dataloader):
        img_list = []
        self.G_losses = []
        self.D_losses = []
        iters = 0

        fixed_noise = torch.randn(self.batch_size, 100)
        self.train()
        for e in range(0, epoch):
            train_loss = 0
            for i, (x, y) in enumerate(dataloader):
                self.discriminator.zero_grad()

                # train Discriminator with real data
                dis_out = self.discriminator(x)
                label = torch.ones_like(dis_out) # 1 : real label
                errD_real = self.loss(dis_out, label)
                errD_real.backward()
                D_x = dis_out.data.mean()

                # train Discriminator with fake data
                noise = torch.randn(self.batch_size, 100)
                gen_out = self.generator(noise)
                """"!!! discriminator 학습시 generator는 학습해서는 안된다!! 따라서 detach()를 사용해서 generator의 파라미터가 관여 못하게!"""
                dis_out_fake = self.discriminator(gen_out.detach())
                label = torch.zeros_like(dis_out_fake) # fake label
                errD_fake = self.loss(dis_out_fake, label)
                errD_fake.backward()
                D_G_z1 = dis_out_fake.data.mean()

                errD = errD_real + errD_fake
                train_loss += errD.item()
                self.optimizerD.step()


                # train generator
                self.generator.zero_grad()
                label = torch.ones_like(dis_out_fake) # real label to train generator
                output = self.discriminator(gen_out)

                errG = self.loss(output, label)
                errG.backward()
                D_G_z2 = output.data.mean()
                self.optimizerG.step()


                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (e, epoch, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # 이후 그래프를 그리기 위해 손실값들을 저장해둡니다
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

                # fixed_noise를 통과시킨 G의 출력값을 저장해둡니다
                if (iters % 500 == 0) or ((e == epoch - 1) and (i == len(dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

        self.showLossGraph()


    def generate(self, z):
        generated_image = self.generator(z)
        return generated_image

    def showLossGraph(self):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()








