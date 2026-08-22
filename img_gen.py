import torch
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from torchvision.utils import save_image
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = 'Exoplanet Images'

transform_img = Compose([
    torchvision.transforms.Resize((32, 32)),  # Resize to 256x256
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
])

image_data = torchvision.datasets.ImageFolder(root=data_path, transform=transform_img)

batch_size = 32
data_loader = DataLoader(image_data, batch_size=batch_size, shuffle=True, num_workers=0)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Generator Network
image_size = 512 * 512 * 3
hidden_size = 128
latent_size = 100  # Change latent_size to 100
channels = 3
image_channels = 3


# Weight initialization function
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)


Discriminator = nn.Sequential(
    nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(128 * 4 * 4, 256),  # Change the last linear layer input size to 256
    nn.ReLU(),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

Discriminator.to(device)
Discriminator.apply(weights_init)  # Apply weight initialization to the Discriminator

Generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, hidden_size * 4, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(hidden_size * 4),
    nn.ReLU(True),

    nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(hidden_size * 2),
    nn.ReLU(True),

    nn.ConvTranspose2d(hidden_size * 2, hidden_size, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(hidden_size),
    nn.ReLU(True),

    nn.ConvTranspose2d(hidden_size, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
)

Generator.to(device)
Generator.apply(weights_init)  # Apply weight initialization to the Generator

# Loss and Optimizers
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(Discriminator.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(Generator.parameters(), lr=0.0002)

# Feature Matching Loss
feature_criterion = nn.L1Loss()  # Use L1 loss for feature matching


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


def train_discriminator(images):
    # Create the labels for real and fake images
    batch_size = images.size(0)
    real_labels = torch.ones(batch_size, 1).to(device) * 0.9  # Use label smoothing for real labels
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # Generate fake images
    z = torch.randn(batch_size, latent_size, 1, 1).to(device)  # Adjust the generator input shape
    fake_images = Generator(z)

    # Loss for real images
    outputs_real = Discriminator(images)
    d_loss_real = criterion(outputs_real, real_labels[:batch_size])  # Use the same batch size for real labels
    real_score = outputs_real.mean().item()

    # Loss for fake images
    outputs_fake = Discriminator(fake_images.detach())
    d_loss_fake = criterion(outputs_fake, fake_labels[:batch_size])  # Use the same batch size for fake labels
    fake_score = outputs_fake.mean().item()

    # Feature Matching Loss
    real_features = Discriminator[:7](images)  # Update the range to match the feature layer count
    fake_features = Discriminator[:7](fake_images)  # Update the range to match the feature layer count
    feature_loss = feature_criterion(real_features, fake_features.detach())

    # Combine losses
    d_loss = d_loss_real + d_loss_fake + feature_loss
    reset_grad()
    d_loss.backward()
    d_optimizer.step()

    return d_loss, real_score, fake_score


def train_generator():
    # Generate fake images and calculate loss
    batch_size = 64  # Set the batch size for fake images
    z = torch.randn(batch_size, latent_size, 1, 1).to(device)  # Adjust the generator input shape
    fake_images = Generator(z)
    labels = torch.ones(batch_size, 1).to(device)
    g_loss = criterion(Discriminator(fake_images), labels)  # Use the same batch size for fake labels

    # Backprop and optimize
    reset_grad()
    g_loss.backward()
    g_optimizer.step()
    return g_loss, fake_images


sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


def save_fake_images(index):
    num_samples = 100
    sample_vectors = torch.randn(num_samples, latent_size, 1, 1).to(device)  # Adjust the generator input shape
    fake_images = Generator(sample_vectors)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('Saving', fake_fname)

    # Denormalize and save images
    fake_images = denormalize(fake_images)
    save_image(fake_images, os.path.join(sample_dir, fake_fname), nrow=10, padding=2)


def img_runner():
    num_epochs = 400
    total_step = len(data_loader)
    d_losses, g_losses, real_scores, fake_scores = [], [], [], []

    # Set the models to training mode
    Generator.train()
    Discriminator.train()

    # Variables to store the "least fake" and "best real" images and their scores
    least_fake_image, least_fake_score = None, float('inf')
    best_real_image, best_real_score = None, float('-inf')

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device)

            # Train the discriminator and generator
            d_loss, real_score, fake_score = train_discriminator(images)
            g_loss, fake_images = train_generator()

            # Update "best real" image and its score
            if real_score > best_real_score:
                best_real_score = real_score
                best_real_image = images[0].cpu()

            # Inspect the losses
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            real_scores.append(real_score)
            fake_scores.append(fake_score)
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch+1, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                          real_score, fake_score))

        # Save fake images
        save_fake_images(epoch + 1)

    # Find the best combination of "least fake" and "best real" images based on their discriminator scores
    overall_real_image = torch.cat([best_real_image], dim=2)

    # Save the best combination image
    save_image(denormalize(overall_real_image), 'best_overall_image.png')

    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
    plt.plot(g_losses, label='Generator Loss', alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Discriminator and Generator Loss')
    plt.show()

    # Save the trained models
    torch.save(Generator.state_dict(), 'G.ckpt')
    torch.save(Discriminator.state_dict(), 'D.ckpt')


# Call img_runner after the training loop
# img_runner()
