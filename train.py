# Acknowledgment: The code is modified based on https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import numpy as np  # to handle matrix and data operation
import os  # operation system library
import torch  # to load pytorch library
import torch.nn as nn  # to load pytorch library
from model import Generator, Discriminator  # load network structures from model.py
import torch.utils.data  # to load data processor
from torch.autograd import Variable  # pytorch data type
from torchvision import datasets, transforms  # to load torch data processor
import argparse  # it is a library to control parameters
import time  # it is a library to get current time
import torch.optim as optim # load optimization function
import torchvision.utils as vutils # a library of pytorch 
import matplotlib.pyplot as plt # to visualize the result
import matplotlib.animation as animation # it is a library for visualization intermediate results
import random # to generate random latent vectors

# ============================ parse the command line =================================
parser = argparse.ArgumentParser()
parser.add_argument("-Batch_size", type=int, default=64, help="Batch number")
parser.add_argument("-Epoch", type=int, default=300, help="Number of epochs")
parser.add_argument("-lr", type=float, default=0.0002, help="Learning rate")
parser.add_argument("-isUseGPU", type=bool, default=True, help="use GPU (1) or CPU (0)")

opt = parser.parse_args()  # zip all above parameter to a variable "opt"

# ============================ load training and testing data to torch data loader===============
BATCH_SIZE = opt.Batch_size  # get the batch size from the zipped parameters
IMAGE_SIZE = 256
l1_coef = 1 #If image too blurry, increase l1 coeff and let image sharper.
l2_coef = 1 #If image too noisy or artifacts, increase l2 coeff and smooth them out

resume_training = True  # Set to True if you want to resume training from a checkpoint

dataset_train = datasets.ImageFolder(root="./dataset",
                                     transform=transforms.Compose([
                                       transforms.Resize(IMAGE_SIZE),
                                         transforms.CenterCrop(IMAGE_SIZE),
                                         transforms.ToTensor(),
                                     ]))
train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=4,
                                           drop_last=True)  # call DataLoader to separate data into a number of batches. test_loader contains the test batches

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.isUseGPU) else "cpu")
print("Use %s for training" % device)

# Define training settings
num_epochs = opt.Epoch
netD = Discriminator().to(device)
netG = Generator().to(device)
criterion = nn.BCELoss()  # Initialize Binary Cross Entropy loss (BCELoss) function
l2_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))  # Setup Adam optimizers for Discriminator
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))  # Setup Adam optimizers for Generator

real_label = 1.
fake_label = 0.

# Logger
G_losses = []
D_losses = []
iters = 0
img_list = []

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(opt.Batch_size, 100, 1, 1, device=device)

# Define useful functions
def savecheckpoint(epoch, netG, netD):  # save check point files
    model_out_path_G = "models/Training_epoch_{}_G.pth".format(epoch)  # define the name of learned model
    torch.save(netG.state_dict(), model_out_path_G)  # save learned model
    print("Checkpoint of Generator is saved to {}".format(model_out_path_G))
    model_out_path_D = "models/Training_epoch_{}_D.pth".format(epoch)  # define the name of learned model
    torch.save(netD.state_dict(), model_out_path_D)  # save learned model
    print("Checkpoint of Discriminator is saved to {}".format(model_out_path_D))
    return

def load_checkpoint(epoch, netG, netD, optimizerG, optimizerD):
    # Define the paths to the generator and discriminator model checkpoints
    model_path_G = "models/Training_epoch_{}_G.pth".format(epoch)
    model_path_D = "models/Training_epoch_{}_D.pth".format(epoch)
    # Load the state dictionaries for the generator and discriminator from the checkpoints
    netG.load_state_dict(torch.load(model_path_G))
    netD.load_state_dict(torch.load(model_path_D))
    print("Loaded checkpoint from epoch {}".format(epoch))

# Load the model if resuming training
start_epoch = 0

resume_training = True  # Set to True if you want to resume training from a checkpoint

# Check if we need to resume training from a specific epoch
if resume_training:
    start_epoch = 249  # Set this to the epoch you want to resume from
    load_checkpoint(start_epoch, netG, netD, optimizerG, optimizerD)
    

    
print("Starting Training Loop...")
for epoch in range(start_epoch, num_epochs):
    netG.train()
    netD.train()
    # For each batch in the dataloader
    for i, data in enumerate(train_loader, 0):

        ############################
        # (1) Update D network: minimize -(log(D(x)) + log(1 - D(G(z))))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_imgs = data[0].to(device)
        b_size = real_imgs.size(0)
        
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_imgs).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        
        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, 100, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        errD.backward()
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: minimize -log(D(G(z)))
        ###########################
        netG.zero_grad()
        #Generate new noise
        noise = torch.randn(b_size, 100, 1, 1, device=device)
        # Generate new fake image batch with G
        fake_imgs = netG(noise)
        
        # Get discriminator output for the new fake images
        out_fake = netD(fake_imgs)
        
        # Get discriminator output for the real images
        out_real = netD(real_imgs)
        
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        
        # Calculate G's loss based on this output
        errG_BCE = criterion(output, label)
        errG_l1 = l1_coef * l1_loss(fake_imgs, real_imgs)
        errG_l2 = l2_coef * l2_loss(torch.mean(out_fake,0), torch.mean(out_real,0).detach())
        
        #Compute total generator loss
        errG = errG_BCE + errG_l1 + errG_l2
        
        # Calculate gradients for G
        errG.backward()
        # Update G
        optimizerG.step()
        # adding loss to the list
        D_losses.append(errD.item())
        G_losses.append(errG.item())
        
        
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, num_epochs, i, len(train_loader),
                     errD.mean().item(), errG.mean().item()))
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 10 == 0) or ((epoch == num_epochs - 1) and (i == len(train_loader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    
    # Generate and save images every 10 epochs
    if (epoch + 1) % 5 == 0:
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        writervideo = animation.FFMpegWriter(fps=5, codec='libx264')
        ani.save(f'FIN_FIN_train_progress_epoch_{epoch + 1}.mp4', writer=writervideo)

        # Clear img_list for the next epoch
        img_list.clear()
    
    
    # Clear cache to free up memory
    torch.cuda.empty_cache()

print("Completed")
# visualize and save the training loss as an image
plt.figure(figsize=(10,5))
plt.title("Generator Loss During Training")
plt.plot(G_losses)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.show()

plt.savefig(os.path.join("results", 'output_generatorLoss.png'))

plt.figure(figsize=(10,5))
plt.title("Discriminator Loss During Training")
plt.plot(D_losses)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.show()

plt.savefig(os.path.join("results", 'output_discriminatorLoss.png'))
