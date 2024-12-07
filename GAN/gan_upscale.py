# adds parent directory to python path to import other utility modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from data.cifar_10_data import CIFAR10Data
from torch.utils.data import DataLoader, TensorDataset
import os
from data.diverse_image_net_data import DiverseImageNetData
import random

# Define the ResidualBlock as before
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.conv_block(x)

# Define the UpsampleBlock as before
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)

# Updated Generator Class to accept 3 x 64 x 64 input images and output 3 x 224 x 224 images without using nn.Upsample
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Initial convolution block
        self.initial_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        
        # 16 Residual Blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]
        )
        
        # Upsampling blocks: two UpsampleBlocks with scale factor 2
        # This will upscale from 64x64 -> 128x128 -> 256x256
        self.upsampling_blocks = nn.Sequential(
            UpsampleBlock(64, 2),  # 64 -> 128
            UpsampleBlock(64, 2),  # 128 -> 256
        )
        
        # Output block
        self.output_block = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        # Initial convolution
        initial = self.initial_block(x)
        
        # Pass through residual blocks
        residual = self.residual_blocks(initial)
        
        # Add skip connection
        upsampled = self.upsampling_blocks(residual + initial)
        
        # Crop the center 224x224 region from 256x256
        # Calculate the amount to crop: (256 - 224) / 2 = 16 pixels on each side
        crop_size = 16
        upsampled = upsampled[:, :, crop_size:-crop_size, crop_size:-crop_size]
        
        # Generate output image
        return self.output_block(upsampled)


# Discriminator: Distinguishes between real and generated high-res images
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Use dummy input to calculate fc1 input size
        dummy_input = torch.randn(1, 3, 224, 224)
        conv_output_size = self.model(dummy_input).view(-1).size(0)

        self.fc1 = nn.Linear(conv_output_size, 1024)  # Dynamically calculated size
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return self.sigmoid(x)



if __name__ == "__main__":

    # dummy_input = torch.randn(1, 3, 224, 224)  # Example input: Batch size 1, 3x224x224
    # discriminator = Discriminator()
    # output = discriminator.model(dummy_input)
    # print(output.shape)  # Inspect the shape
    # exit()
    NUM_EPOCHS = 250
    BATCH_SIZE = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader = DiverseImageNetData.gen_data(batch_size=BATCH_SIZE)

    criterion_adv = nn.BCEWithLogitsLoss()  # For discriminator classification
    criterion_pixel = nn.MSELoss()  # For pixel-wise reconstruction

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # Train Discriminator
            optimizer_d.zero_grad()

            # Real images
            real_labels = torch.ones(hr_imgs.size(0), 1, device=device)  # Add dimension
            real_outputs = discriminator(hr_imgs)
            d_real_loss = criterion_adv(real_outputs, real_labels)

            # Fake images
            fake_imgs = generator(lr_imgs)
            fake_labels = torch.zeros(hr_imgs.size(0), 1, device=device)  # Add dimension
            fake_outputs = discriminator(fake_imgs.detach())
            d_fake_loss = criterion_adv(fake_outputs, fake_labels)

            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()

            # Adversarial loss (generator wants discriminator to classify fake as real)
            adv_loss = criterion_adv(discriminator(fake_imgs), real_labels)

            # Pixel-wise reconstruction loss
            pixel_loss = criterion_pixel(fake_imgs, hr_imgs)

            # Total generator loss
            g_loss = adv_loss + 0.001 * pixel_loss  # Adjust weight of pixel loss as needed
            g_loss.backward()
            optimizer_g.step()

            # Print batch progress
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                    f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        # Save generated images periodically
        if epoch % 5 == 0:
            with torch.no_grad():
                # Function to sample random images from a DataLoader
                def sample_random_images(loader, num_samples=4):
                    dataset = loader.dataset
                    sampled_indices = random.sample(range(len(dataset)), num_samples)
                    sampled_lr = torch.stack([dataset[i][0] for i in sampled_indices]).to(device)
                    sampled_hr = torch.stack([dataset[i][1] for i in sampled_indices]).to(device)
                    return sampled_lr, sampled_hr

                # Sample from Training Data
                lr_train_imgs, hr_train_imgs = sample_random_images(train_loader, num_samples=4)
                fake_train_imgs = generator(lr_train_imgs).cpu()

                # Sample from Validation Data
                lr_val_imgs, hr_val_imgs = sample_random_images(val_loader, num_samples=4)
                fake_val_imgs = generator(lr_val_imgs).cpu()

                # Denormalize images for display (from [-1, 1] to [0, 1])
                def denormalize(tensor):
                    return (tensor + 1) / 2.0

                lr_train_imgs_vis = denormalize(lr_train_imgs.cpu())
                fake_train_imgs_vis = denormalize(fake_train_imgs)
                hr_train_imgs_vis = denormalize(hr_train_imgs.cpu())

                lr_val_imgs_vis = denormalize(lr_val_imgs.cpu())
                fake_val_imgs_vis = denormalize(fake_val_imgs)
                hr_val_imgs_vis = denormalize(hr_val_imgs.cpu())

                # Create the plot
                num_samples = 4
                fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4 * num_samples))

                for i in range(num_samples):
                    # ===== Training Data =====
                    # Low-Resolution Image
                    axes[i, 0].imshow(lr_train_imgs_vis[i].permute(1, 2, 0).numpy())
                    axes[i, 0].set_title("Train Low-Res")
                    axes[i, 0].axis("off")

                    # Generated High-Resolution Image
                    axes[i, 1].imshow(fake_train_imgs_vis[i].permute(1, 2, 0).numpy())
                    axes[i, 1].set_title("Train Generated High-Res")
                    axes[i, 1].axis("off")

                    # True High-Resolution Image
                    axes[i, 2].imshow(hr_train_imgs_vis[i].permute(1, 2, 0).numpy())
                    axes[i, 2].set_title("Train True High-Res")
                    axes[i, 2].axis("off")

                    # ===== Validation Data =====
                    # Low-Resolution Image
                    axes[i, 3].imshow(lr_val_imgs_vis[i].permute(1, 2, 0).numpy())
                    axes[i, 3].set_title("Val Low-Res")
                    axes[i, 3].axis("off")

                    # Generated High-Resolution Image
                    axes[i, 4].imshow(fake_val_imgs_vis[i].permute(1, 2, 0).numpy())
                    axes[i, 4].set_title("Val Generated High-Res")
                    axes[i, 4].axis("off")

                    # True High-Resolution Image
                    axes[i, 5].imshow(hr_val_imgs_vis[i].permute(1, 2, 0).numpy())
                    axes[i, 5].set_title("Val True High-Res")
                    axes[i, 5].axis("off")

                fig.suptitle(f'Resolution Upscaling GAN Performance at Epoch {epoch}', fontsize=24)

                # Adjust layout to reduce white space
                plt.tight_layout(pad=2.0)  # Adjust padding
                fig.subplots_adjust(top=0.9)  # Optional: Fine-tune the top margin

                # Save the figure with tight bounding box
                os.makedirs("GAN/upscale_results", exist_ok=True)
                plt.savefig(f"GAN/upscale_results/comparison_epoch_{epoch}.png", bbox_inches='tight')
                plt.close()