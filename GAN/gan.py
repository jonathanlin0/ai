import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(noise_dim, 7 * 7 * 256),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)
    
# Discriminator
class Discriminator(nn.Module):
    """
    This discriminator is for MNIST dataset only. Assumes a 28x28 input
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(7 * 7 * 128, 1)
        )

    def forward(self, x):
        return self.main(x)
    
if __name__ == "__main__":
    NOISE_DIM = 100
    NUM_EXAMPLES = 16

    generator = Generator(NOISE_DIM)
    discriminator = Discriminator()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    criterion = nn.BCEWithLogitsLoss()

    generator_optimizer = optim.Adam(generator.parameters(), lr = 0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    NUM_EPOCHS = 100
    BATCH_SIZE = 256

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
            real_loss.backward()

            # train discriminator with fake images
            noise = torch.randn(real_images.size(0), NOISE_DIM, device=device)
            fake_images = generator(noise)
            fake_labels = torch.zeros(real_images.size(0), 1, device=device)
            fake_outputs = discriminator(fake_images.detach())
            fake_loss = criterion(fake_outputs, fake_labels)
            fake_loss.backward()
            discriminator_optimizer.step()

            # train generator
            generator_optimizer.zero_grad()
            fake_labels = torch.ones(real_images.size(0), 1, device=device)
            fake_outputs = discriminator(fake_images)
            gen_loss = criterion(fake_outputs, fake_labels)
            gen_loss.backward()
            generator_optimizer.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], '
                  f'Discriminator Loss: {real_loss.item() + fake_loss.item():.4f}, '
                  f'Generator Loss: {gen_loss.item():.4f}')
    
    def generate_and_save_images(model, noise):
        model.eval()

        with torch.no_grad():
            fake_images = model(noise).cpu()
            fake_images = fake_images.view(fake_images.size(0), 28, 28)

            # we're assuming 16 samples to be generated. change the figure dimensions as needed
            fig = plt.figure(figsize=(4, 4))
            for i in range(fake_images.size(0)):
                plt.subplot(4, 4, i+1)
                plt.imshow(fake_images[i], cmap="gray")
                plt.axis("off")
            
            plt.savefig("GAN/MNIST_gen.png")
            plt.show()
    
    test_noise = torch.randn(NUM_EXAMPLES, NOISE_DIM, device=device)
    generate_and_save_images(generator, test_noise)