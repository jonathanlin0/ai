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


from data.image_net_data import ImageNetData

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = self.relu(out)
        return out


# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 7 * 7 * 256),  # Change output shape to match 7x7 feature map
            nn.ReLU(True)
        )

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(3)]  # Adjust number of channels
        )

        # Upsampling blocks
        self.upsample_blocks = nn.Sequential(
            # Upsample: 7x7 -> 14x14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ResidualBlock(128),

            # Upsample: 14x14 -> 28x28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            ResidualBlock(64),
        )

        # Final convolution layer
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),  # Output 1 channel
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 7, 7)  # Reshape to feature map of 7x7
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = self.upsample_blocks(x)
        x = self.conv_out(x)
        return x


# Discriminator
class Discriminator(nn.Module):
    """
    Discriminator for MNIST (1x28x28 images).
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 7x7 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),

            nn.Flatten(),  # Flatten feature map
            nn.Linear(4 * 4 * 256, 1)  # Adjusted to match 4x4 feature map
        )

    def forward(self, x):
        return self.main(x)

def generate_and_save_images(model, noise, epoch, save_dir):
    # Ensure the provided directory exists
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # Generate images using the generator
        fake_images = model(noise).cpu()

        # Plot the generated images
        fig = plt.figure(figsize=(4, 4))
        for i in range(fake_images.size(0)):
            plt.subplot(4, 4, i + 1)

            # Transpose image dimensions from [1, 28, 28] to [28, 28] for visualization (MNIST case)
            img = fake_images[i, 0].numpy()

            # Normalize pixel values to [0, 1] for display
            img = (img + 1) / 2.0  # Assuming Tanh was used as activation

            # Plot the image
            plt.imshow(img, cmap="gray")
            plt.axis("off")

        # Save the grid of images as a single file
        plt.savefig(os.path.join(save_dir, f"generated_mnist_grid_{epoch}.png"))

    
if __name__ == "__main__":
    torch.manual_seed(42)

    NOISE_DIM = 128
    NUM_EXAMPLES = 16

    generator = Generator(NOISE_DIM)
    discriminator = Discriminator()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    criterion = nn.BCEWithLogitsLoss()

    generator_optimizer = optim.Adam(generator.parameters(), lr = 0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    NUM_EPOCHS = 25
    BATCH_SIZE = 32

    # create the dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data/datasets', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(train_loader):
            real_images, _ = data
            real_images = real_images.to(device)

            # train discriminator with real images
            discriminator_optimizer.zero_grad()
            real_labels = torch.ones(real_images.size(0), 1, device=device)
            real_outputs = discriminator(real_images)
            real_loss = criterion(real_outputs, real_labels)
            # .backward() calcs the gradient values for the discriminator
            real_loss.backward()

            # train discriminator with fake images
            noise = torch.randn(real_images.size(0), NOISE_DIM, device=device)
            fake_images = generator(noise)
            fake_labels = torch.zeros(real_images.size(0), 1, device=device)
            fake_outputs = discriminator(fake_images.detach())
            fake_loss = criterion(fake_outputs, fake_labels)
            # .backward() calcs the gradient values for the discriminator
            fake_loss.backward()
            # updates the discriminator based on the stored gradients
            discriminator_optimizer.step()

            # train and update generator
            generator_optimizer.zero_grad()
            # use labels = 1 cause the generator wants to trick the discriminator. so discriminator guessing 1 on the fake images is incorrect, but good for the generator
            fake_labels = torch.ones(real_images.size(0), 1, device=device)
            fake_outputs = discriminator(fake_images)
            gen_loss = criterion(fake_outputs, fake_labels)
            # calculate the gradient values for the generator
            gen_loss.backward()
            generator_optimizer.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], '
                  f'Discriminator Loss: {real_loss.item() + fake_loss.item():.4f}, '
                  f'Generator Loss: {gen_loss.item():.4f}')
        if epoch % 5 == 0:
            noise = torch.randn(NUM_EXAMPLES, NOISE_DIM, device=device)
            generate_and_save_images(generator, noise, epoch, "GAN/generated_mnist")
            generator.train()
    noise = torch.randn(NUM_EXAMPLES, NOISE_DIM, device=device)
    generate_and_save_images(generator, noise, epoch, "GAN/generated_mnist")
