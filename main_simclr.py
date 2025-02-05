import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torchvision.models import resnet18
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import os
import umap




# ---- SimCLR Data Augmentation ----
class SimCLRTransform:
    def __init__(self, num_channels=3, is_classification=False):
        """
        Defines transformations for contrastive learning (SimCLR) and classification.
        
        Args:
            num_channels (int): 1 for grayscale (e.g., MNIST), 3 for RGB (e.g., CIFAR).
            is_classification (bool): If True, returns single images for classification.
        """
        self.is_classification = is_classification
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=num_channels) if num_channels == 3 else transforms.Lambda(lambda x: x),
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __call__(self, x):
        """
        Returns either:
        - Two augmented images (for contrastive learning)
        - A single augmented image (for classification)
        """
        if self.is_classification:
            return self.transform(x)  # Single transformed image for classification
        return self.transform(x), self.transform(x)  # Two views for contrastive learning


# ---- Dataset Loader ----
def get_dataset(name, train=True, is_classification=False):
    """
    Returns the specified dataset with appropriate transformations.

    Args:
        name (str): Dataset name ('mnist', 'cifar10', 'cifar100').
        train (bool): Whether to load the training split (True) or test split (False).
        is_classification (bool): If True, returns single images for classification.

    Returns:
        torchvision dataset object
    """
    num_channels = 1 if name == 'mnist' else 3
    transform = SimCLRTransform(num_channels, is_classification)

    if name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, transform=transform, download=True)
    elif name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=train, transform=transform, download=True)
    elif name == 'mnist':
        dataset = torchvision.datasets.MNIST(root='./data', train=train, transform=transform, download=True)
    else:
        raise ValueError("Unsupported dataset. Choose from 'cifar10', 'cifar100', or 'mnist'.")
    
    return dataset


# ---- Train Classifier ----
def train_classifier(model, dataloader, optimizer, criterion, epochs=20, device='cuda'):
    """
    Trains a linear classifier on top of the pretrained SimCLR encoder.

    Args:
        model (nn.Module): Linear classifier model.
        dataloader (DataLoader): Dataloader for training the classifier.
        optimizer (torch.optim): Optimizer for training.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        epochs (int): Number of training epochs.
        device (str): 'cuda' or 'cpu'.
    """
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for x, y in dataloader:
            x = x.to(device)  # Ensure x is a tensor
            y = y.to(device)  # Ensure y is a tensor
            
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}, Acc: {correct / total:.4f}")


# ---- Save & Load Functions ----
def save_encoder(model, path="simclr_encoder.pth"):
    torch.save(model.encoder.state_dict(), path)
    print(f"SimCLR encoder saved to {path}")

def load_encoder(model, path="simclr_encoder.pth"):
    model.encoder.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.encoder.eval()
    print(f"SimCLR encoder loaded from {path}")



# ---- NT-Xent Loss for SimCLR ----
def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]
    z = torch.cat((z_i, z_j), dim=0)
    similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)

    logits = similarity_matrix / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    loss = -torch.sum(labels * log_prob) / (2 * batch_size)
    return loss


# ---- SimCLR Model ----
class SimCLR(nn.Module):
    def __init__(self, base_encoder=resnet18, projection_dim=128, input_channels=3):
        super(SimCLR, self).__init__()
        # self.encoder = base_encoder(pretrained=False)
        self.encoder = base_encoder(weights=None) if hasattr(base_encoder, "weights") else base_encoder(pretrained=False)


        if input_channels == 1:
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder.fc = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim, affine=False)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z


# ---- Linear Classifier ----
class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.encoder = encoder
        self.encoder.eval()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.fc(features)


# ---- Pretrain SimCLR ----
def pretrain_simclr(model, dataloader, optimizer, epochs=5, device='cuda'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (x_i, x_j), _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_i, x_j = x_i.to(device), x_j.to(device)
            _, z_i = model(x_i)
            _, z_j = model(x_j)

            loss = nt_xent_loss(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

    save_encoder(model)


# ---- Test Classifier ----
def test_classifier(model, dataloader, device='cuda'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")


# ---- Visualization with UMAP ----
def visualize_latent_space_umap(encoder, dataloader, device, save_path=None):
    encoder.eval()
    latent_vectors, labels_list = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            latent_representations = encoder.encoder(images)

            latent_vectors.append(latent_representations.cpu().numpy())
            labels_list.append(labels.numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_results = reducer.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels_list, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("UMAP Visualization of Latent Space")

    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


# ---- Main Function ----
def main():
    dataset_name = "mnist"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = get_dataset(dataset_name, train=True)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    model = SimCLR(input_channels=1 if dataset_name == "mnist" else 3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    pretrain_simclr(model, train_loader, optimizer, epochs=1, device=device)

    # Now train classifier
    classifier = LinearClassifier(model.encoder).to(device)
    classifier_dataset = get_dataset(dataset_name, train=True, is_classification=True)
    classifier_dataset_loader = DataLoader(classifier_dataset, batch_size=512, shuffle=True)
    train_classifier(classifier, classifier_dataset_loader, optimizer, nn.CrossEntropyLoss(), epochs=2, device=device)

    test_dataset = get_dataset(dataset_name, train=False, is_classification=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    test_classifier(classifier, test_loader, device=device)

    visualize_latent_space_umap(classifier.encoder, test_loader, device, save_path="./umap.png")


if __name__ == "__main__":
    main()
