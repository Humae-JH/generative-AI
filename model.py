from torch import nn
import torch
from abc import *
from torch.optim import *
import torchvision
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt
import numpy as np
import copy
import numpy as np
import math
import random
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
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

    def showImage(self, images, title = "image"):
        img_grid = vutils.make_grid(images, padding=2, normalize=True)
        # Show the images
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.title(title)
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

    def saveState(self, model, model_name):
        if ".pth" not in model_name:
            model_name = model_name + ".pth"
        torch.save(model.state_dict(), model_name)
        print(f"weight has been saved [{model_name}]")
        return

    def loadState(self, model, path):
        if ".pth" not in path:
            model = path + ".pth"
        model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"{model}'s weight has been loaded [{path}]")
        return



class VAEEncoder(BaseModel):
    def __init__(self, device, lr):
        super().__init__(device, lr)
        self.network = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1), # 64->32
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, 3, 2, 1), # 32 -> 16
                nn.LeakyReLU(),
                nn.Conv2d(128, 256, 3 , 2, 1), #16 -> 8
                nn.LeakyReLU(),
                nn.Conv2d(256, 512, 3, 2, 1), # 4
                nn.LeakyReLU(),
                nn.Conv2d(512, 1024, 4, 1, 0), # 1
                nn.Flatten(1, -1)
        )

        self.z_mean = nn.Linear(1024, 100)
        self.z_log_var = nn.Linear(1024, 100)

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
        self.linear = nn.Linear(100, 2048)

        self.network = nn.Sequential(
            nn.Unflatten(1, torch.Size([2048,1,1])),
            nn.ConvTranspose2d(2048,1024, 4,1, 0), # 1->4
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512,  4, 2, 1), # 4 -> 8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 8 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 16 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, 2, 1), # 32 -> 64
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
        self.Reconloss = torch.nn.L1Loss()

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
        ).to(self.device)

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
        ).to(self.device)
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        self.optimizerG = Adam(self.generator.parameters(), self.learning_rate, betas=(0.5, 0.999))
        self.optimizerD = Adam(self.discriminator.parameters(), self.learning_rate / 4.0, betas=(0.5, 0.999))

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

        fixed_noise = torch.randn(self.batch_size, 100).to(self.device)
        self.train()
        for e in range(0, epoch):
            train_loss = 0
            for i, (x, y) in enumerate(dataloader):
                self.discriminator.zero_grad()
                x = x.to(self.device)
                # train Discriminator with real data
                dis_out = self.discriminator(x)
                label = torch.ones_like(dis_out) # 1 : real label
                errD_real = self.loss(dis_out, label)
                errD_real.backward()
                D_x = dis_out.data.mean()

                # train Discriminator with fake data
                noise = torch.randn(self.batch_size, 100).to(self.device)
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


class UNet(BaseModel):
    def __init__(self, device, lr):
        super().__init__(device, lr)



        self.down1_res1 = ResidualBlock(3, 16)
        self.down1_res2 = ResidualBlock(16, 32)
        self.avgPool = nn.AvgPool2d(2)

        self.down2_res1 = ResidualBlock(32, 48)
        self.down2_res2 = ResidualBlock(48, 64)

        self.down3_res1 = ResidualBlock(64, 80)
        self.down3_res2 = ResidualBlock(80, 96)

        self.bottomRes1 = ResidualBlock(96, 128)
        self.bottomRes2 = ResidualBlock(128, 96)

        self.upSampling = nn.Upsample(scale_factor=2, mode="nearest")
        self.up1_res1 = ResidualBlock(192, 80)
        self.up1_res2 = ResidualBlock(80, 64)

        self.up2_res1 = ResidualBlock(128, 48)
        self.up2_res2 = ResidualBlock(48, 32)

        self.up3_res1 = ResidualBlock(64, 16)
        self.up3_res2 = ResidualBlock(16, 3)

        self.optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        self.loss = nn.L1Loss()
        self.normalizer = nn.LayerNorm([3,32,32])
    def forward(self,x):
        # Downblock 1
        out = self.down1_res1(x)
        out = self.down1_res2(out)
        skip1 = torch.clone(out)
        out = self.avgPool(out)

        # Downblock 2
        out = self.down2_res1(out)
        out = self.down2_res2(out)
        skip2 = torch.clone(out)
        out = self.avgPool(out)

        # Downblock3
        out = self.down3_res1(out)
        out = self.down3_res2(out)
        skip3 = torch.clone(out)
        out = self.avgPool(out)

        # bottomblock
        out = self.bottomRes1(out)
        out = self.bottomRes2(out)

        # Upblock1
        out = self.upSampling(out)
        out = torch.cat((skip3, out), 1) # skip connection from Downblock 3
        out = self.up1_res1(out)
        out = self.up1_res2(out)

        #Upblock2
        out = self.upSampling(out)
        out = torch.cat((skip2, out), 1)
        out = self.up2_res1(out)
        out = self.up2_res2(out)

        #Upblock 3
        out = self.upSampling(out)
        out = torch.cat((skip1, out), 1)
        out = self.up3_res1(out)
        out = self.up3_res2(out)
        out = self.normalizer(out)
        return out


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

                y_hat = self.forward(x)
                loss = self.loss(y_hat, x)
                train_loss += loss.item()
                if _ % 100 == 0:
                    print(f"train_loss : {train_loss / len(dataloader.dataset)}")
                loss.backward()
                self.optimizer.step()

class UNet_Diff(UNet):
    def __init__(self, device, lr):
        super().__init__(device, lr)
        self.down1_res1 = ResidualBlock(4, 16)
        self.SinEmb = SinusoidalPositionalEmbedding(self.device).to(self.device) # positional embedding with time step t
        #self.noiseUpsampling = nn.Upsample(scale_factor = 4, mode="nearest")

        self.to(self.device)

    def forward(self, x):
        image, t = x
        image = image.to(self.device)
        t = t.to(self.device)
        noise_embedding = self.SinEmb(t, image.shape[2], image.shape[3])
        #noise_embedding = noise_embedding.repeat((1, 1, 1, 1))

        concat_x = torch.concat((image, noise_embedding), dim=1)
        return super().forward(concat_x)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, device):
        self.device = device
        super(SinusoidalPositionalEmbedding, self).__init__()

    def forward(self, t, height, width):
        """
        t: [batchsize, 1, 1, 1]
        """
        batch_size = t.size(0)

        # Create positional encodings for the given time step
        position = t.view(batch_size, 1)  # [batchsize, 1]
        div_term = torch.exp(
            torch.arange(0, 1, 2).float() * (-torch.log(torch.tensor(10000.0)) / 1)).to(self.device)
        pos_embedding = torch.zeros(batch_size, 1).to(self.device)

        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)

        # Reshape and expand to match the target size
        pos_embedding = pos_embedding.view(batch_size, 1, 1, 1)
        pos_embedding = pos_embedding.expand(batch_size, 1, height, width)

        return pos_embedding


class DiffusionModel(BaseModel):
    def __init__(self, device, lr, T = 1000):
        super().__init__(device, lr)
        self.normalizer = nn.BatchNorm2d(3)
        self.network = UNet_Diff(device, lr)
        self.ema_network = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.parameters(), lr= self.learning_rate)
        self.loss = nn.L1Loss()
        self.T = T
        diffusion_times = [x/self.T for x in range(self.T)]
        linear_noise_rates , linear_signal_rates = self.cosine_diffusion_schedule(diffusion_times)
        self.linear_noise_rates = linear_noise_rates
        self.linear_signal_rates = linear_signal_rates

    def getRates(self, diff_time):
        return self.linear_noise_rates[diff_time], self.linear_signal_rates[diff_time]


    def Linear_diffusion_schedule(self, diff_time):
        min_rate = 0.0001
        max_rate = 0.02
        betas = min_rate + torch.tensor(diff_time) * (max_rate - min_rate)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        signal_rates = alpha_bars
        noise_rates = 1 - alpha_bars
        return noise_rates, signal_rates

    def cosine_diffusion_schedule(self, dif_time):
        dif_time = torch.tensor(dif_time, dtype=torch.float32)
        signal_rates = torch.cos(dif_time * torch.pi / 2.)
        noise_rates = torch.sin(dif_time * torch.pi / 2.)
        return noise_rates, signal_rates



    def sigmoid_beta_schedule(self,timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
        """
        sigmoid schedule
        proposed in https://arxiv.org/abs/2212.11972 - Figure 8
        better for images > 64x64, when used during training
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def denoise(self, t, noisy_images, noise_rates, signal_rates, training):
        """if training:
            network = self.network.to(self.device)
        else:
            network = self.ema_network.to(self.device)"""
        network = self.network.to(self.device)

        pred_noises = network.forward([noisy_images, t]) # network get noisy image with noise rates and predict the noise
        pred_images = noisy_images - noise_rates * pred_noises

        return pred_noises, pred_images

    def denormalizer(self, img, data_mean, data_std):
        R_mean , G_mean, B_mean = data_mean
        R_std, G_std, B_std = data_std
        R_mean_array = torch.tensor(R_mean).repeat([img.shape[0],1,img.shape[2], img.shape[3]])
        G_mean_array = torch.tensor(G_mean).repeat([img.shape[0],1,img.shape[2], img.shape[3]])
        B_mean_array = torch.tensor(B_mean).repeat([img.shape[0],1,img.shape[2], img.shape[3]])

        img[0][0] = img[0][0] * R_std + R_mean_array
        img[0][1] = img[0][1] * G_std + G_mean_array
        img[0][2] = img[0][2] * B_std + B_mean_array

        return img

    def forward(self, x):
        pass

    def generate(self, z, data_mean, data_std):
        """ Image Generation Progress
        1. To remove noise from x_t and generate x_t-1, we need to process following progress
            1.1 Predict noise by using x_t -> Diffusion model might learn the noise from x0 to xt with timestep t
            1.2 Estimate x_0 by removing noise ( x_0 = x_t - predicted noise )
            1.3 With x_t and beta(noise schedule), we can add the noise to x_0 and by doing this, we get a x_t-1
            1.4 Repeat 1.1 - 1.3 progress until t = 0
        """
        noise = z
        next_x = z

        for t in range(self.T-1, -1, -1):
            diff_time = torch.tensor(t).reshape(1,1,1,1)
            noise_rates , signal_rates = self.getRates(diff_time)
            noise_rates = noise_rates.to(self.device)
            signal_rates = signal_rates.to(self.device)
            # predict noise added in timestep t
            pred_noise, _ = self.denoise(diff_time, next_x, noise_rates, signal_rates, training=False)

            # Estimate x_0 by removing noise
            est_x_0 = next_x - pred_noise
            if t % 100 == 0:
                img = next_x.detach().cpu()
                img = self.denormalizer(img, data_mean, data_std)
                self.showImage(img, f"{t} / {self.T} reverse image")

            if t > 0:
                # get x_t-1 by adding noise
                next_noise_rates , next_signal_rates = self.getRates(diff_time-1)
                next_noise_rates = noise_rates.to(self.device)
                next_signal_rates = signal_rates.to(self.device)
                next_x = next_signal_rates * est_x_0 + next_noise_rates * noise
            else:
                est_x_0 = next_x - pred_noise
                break


        # need to denormalize
        est_x_0 = self.denormalizer(est_x_0, data_mean, data_std)
        est_x_0 = torch.clip(est_x_0, 0, 1)

        return est_x_0


    def Train(self, epoch, dataloader):
        self.train()
        for e in range(0, epoch):
            for i, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                #x = self.normalizer(x) # normalize images
                #x = torch.nn.functional.normalize(x, dim=1)
                # initiate noises
                noises = torch.nn.functional.normalize(torch.randn([1, 3 , x.shape[2], x.shape[3]]), dim=1).to(self.device)

                batch_size = x.shape[0]
                """diff_time = torch.rand((1, 1, 1, 1)).to(self.device)
                #diff_time = torch.tensor([[[[(i + ((e+1) * len(dataloader))) / len(dataloader) * (e+1)]]]]).to(self.device)
                diff_time = diff_time.repeat((1,1,1,1))
                noise_rates, signal_rates = self.cosine_diffusion_schedule(diff_time)"""

                diff_time = torch.randint(0,self.T,[batch_size,1,1,1])

                noise_rates, signal_rates = self.getRates(diff_time)
                noise_rates = noise_rates.to(self.device)
                signal_rates = signal_rates.to(self.device)

                #signal_rates = torch.sqrt(torch.tensor(1 - (diff_time * (0.001) / self.T)))
                #noise_rates = torch.sqrt(torch.tensor(diff_time * (0.001) / self.T))
                #signal_rates = torch.tensor((self.T - diff_time) / float(self.T))
                #noise_rates = 1 - signal_rates

                # beta need to be fixed!
                #diff_time = torch.tensor(i / len(dataloader)).float().reshape(1,1,1,1).to(self.device)
                #diff_time = torch.rand((batch_size, 1, 1, 1)).to(self.device)

                #noise_rates = diff_time
                #signal_rates = 1 - diff_time


                # create noisy image
                noisy_images = signal_rates * x.detach() + noise_rates * noises

                # predict noise by using noisy image and noise rates
                pred_noises, pred_images = self.denoise(diff_time, noisy_images, noise_rates, signal_rates, training=True)
                """if (i+1) % (len(dataloader) - 1) == 0:
                    #pass
                    self.showImage(noises.to('cpu'), 'noises_image')
                    self.showImage(x.to('cpu'), 'original_images')
                    self.showImage(noisy_images.to('cpu'), f'diff_time : {diff_time} ,noisy_images noise_rates:{noise_rates} ')
                    self.showImage(pred_noises.to('cpu'), 'pred_noises')
                    self.showImage(pred_images.to('cpu'), 'pred_images')"""
                # calculate the noise loss ( -> prediction of noise is our goal )
                #noise_loss = self.loss(pred_noises, noises)
                noise_loss = F.mse_loss(pred_noises, noises, reduction='none')
                noise_loss = reduce(noise_loss, 'b ... -> b', 'mean').sum()

                noise_loss.backward()
                self.optimizer.step()

                if i % 100 == 0:
                    print(f"[epoch : {e} , iteration : {i} / {len(dataloader)}] ..> train loss : {noise_loss} ")

                for weight, ema_weight in zip(self.network.parameters(), self.ema_network.parameters()):
                    ema_weight.data = 0.999 * ema_weight.data + (1 - 0.999) * weight.data

    def setDevice(self, device):
        self.device = device
        self.network.to(torch.device(self.device))
        self.network.SinEmb.to(torch.device(self.device))
        self.network.device = self.device
        self.ema_network.to(torch.device(self.device))
        self.ema_network.SinEmb.to(torch.device(self.device))
        self.ema_network.device = self.device

class ResidualBlock(BaseModel):
    # model which has skip connection for gradient vanishing
    def __init__(self, in_dim, out_dim, device = None, lr = 0):
        super().__init__(device, lr)
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.pixelwiseConv = nn.Conv2d(in_dim, out_dim, kernel_size=1) # revise the channel of input image

        self.normalization = nn.BatchNorm2d(out_dim)
        self.network = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size= 3, padding = 1),
            nn.SiLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self,x):
        if x.shape[1] != self.out_dim:
            x = self.pixelwiseConv(x)
        norm_x = self.normalization(x)
        out = self.network(norm_x)
        out = self.relu(out)
        return out + norm_x # residual








