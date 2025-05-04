# Acknowledgment: The code is modified basd on https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import torch.nn as nn  # to load PyTorch library
import torch  # to load pytorch library

class Generator(nn.Module):  # construct the Generator
    def __init__(self):
        super(Generator, self).__init__()
        # define the network structure
        self.network = nn.Sequential(  # Sequential container
             # Input Size: 100*1*1
            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),


            # State size: 1024 x 4 x 4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),


            # State size: 512 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),


            # State size: 256 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),


            # State size: 128 x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),


            # State size: 64 x 64 x 64
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),


            # State size: 32 x 128 x 128
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # State size: 3 x 256 x 256
        )


    def forward(self, input):
        return self.network(input)

class Discriminator(nn.Module):  # construct the Discriminator
    def __init__(self):
        super(Discriminator, self).__init__()
        # define the network structure
        self.network = nn.Sequential(  # Sequential container
            # input is 3 x 256 x 256
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),

            # state size. 32 x 128 x 128
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # state size. 64 x 64 x 64
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # state size. 128 x 32 x 32
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # state size. 256 x 16 x 16
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # state size. 512 x 8 x 8
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            
            # state size. 1024 x 4 x 4
            nn.Conv2d(1024, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.network(x)
        #print(out.size)
        return out
        


if __name__ == '__main__':
    # Visualize the generator structure
    print("=======================Generator================================")
    m_generator = Generator() # build the generator
    print(m_generator) # display the network structure

    # Visualize the discriminator structure
    print("=======================Discriminator===========================")
    m_discriminator = Discriminator() # build the discriminator
    print(m_discriminator)  # display the network structure