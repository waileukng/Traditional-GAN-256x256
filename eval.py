import numpy as np
import os
import torch
import torch.nn as nn
from model import Generator, Discriminator
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets, transforms
import argparse
import time
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from pytorch_fid import fid_score

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-isUseGPU", type=bool, default=True, help="use GPU (1) or CPU (0)")
parser.add_argument("-G_path", type=str, default="models/Training_epoch_200_G.pth, help="the parameter file of the trained generator")
opt = parser.parse_args()

# Decide which device to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.isUseGPU) else "cpu")
print("Use %s for testing" % device)

# Load the generator
netG = Generator().to(device)
netG.load_state_dict(torch.load(opt.G_path))

# Set the generator to evaluation mode
netG.eval()


# Directories to save real and fake images
real_images_dir = "real_images"
fake_images_dir = "fake_images"

# Create directories if they don't exist
os.makedirs(real_images_dir, exist_ok=True)
os.makedirs(fake_images_dir, exist_ok=True)

# Load real images (replace this with your actual code to load real images)
dataset_real = datasets.ImageFolder(root="./dataset",
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(256),
                                        transforms.ToTensor(),
                                    ]))
real_loader = torch.utils.data.DataLoader(dataset=dataset_real, batch_size=64, shuffle=False)

# Use a fixed noise vector for consistency
random_noise = torch.randn(64, 100, 1, 1).to(device)  # Generate 64 fake images


# Generate and save images
real_imgs = next(iter(real_loader))[0].to(device)  # Load a batch of real images
fake_imgs = netG(random_noise.to(device))  # Generate fake images

# Save real and fake images
for i in range(real_imgs.size(0)):
    vutils.save_image(real_imgs[i], os.path.join(real_images_dir, f"real_{i}.png"))
    vutils.save_image(fake_imgs[i], os.path.join(fake_images_dir, f"fake_{i}.png"))

# Compute FID score
fid_value = fid_score.calculate_fid_given_paths([real_images_dir, fake_images_dir], batch_size=64, device=device, dims=2048)
print(f"FID score: {fid_value}")


def visImg(tensor_img):
    the_first_tensor_img = torch.clone(tensor_img[0].detach()).cpu() # move the first image to host
    plt.imshow(the_first_tensor_img.permute(1, 2, 0)) # use plt to display a tensor image
    plt.show() # display the visualization window
    
def visAllImages(tensor_imgs, nrow=8):
    # Create a grid of images
    grid_img = vutils.make_grid(tensor_imgs, nrow=nrow, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')  # Turn off axis labels
    plt.show()

# Visualize all fake images in one window
visAllImages(fake_imgs)

# Visualize the prediction
#visImg(fake_imgs)