from torch import nn
import torch
from abc import *
from torch.optim import *
import torchvision

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def forward(self,x):
        pass

    def Train(self):
        pass

    def valid(self):
        pass

class VAEEncoder(BaseModel):
    def __init__(self):
        super().__init__()
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
    def __init__(self):
        super().__init__()
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
        super().__init__()
        self.device = device
        self.learning_rate = lr
        self.encoder = VAEEncoder().to(device)
        self.decoder = VAEDecoder().to(device)

        self.optimizer = Adam(self.parameters(), lr=lr)
        #self.KLloss = torch.nn.KLDivLoss()
        self.Reconloss = torch.nn.BCELoss()

    def forward(self, x):
        encoder_out, mu, logvar = self.encoder.forward(x)
        decoder_out = self.decoder.forward(encoder_out)
        return decoder_out

    def Train(self, epoch, dataloader):
        self.train()
        train_loss = 0
        for i in range(0, epoch):

            print(f"[train epoch] : {i} ")
            for  _, (x,y) in enumerate(dataloader):
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


    def inference(self, z):
        self.decoder.forward(z)

class GAN(BaseModel):
    def __init__(self):
        super().__init__()
        pass

    def forward(self,x):
        pass

