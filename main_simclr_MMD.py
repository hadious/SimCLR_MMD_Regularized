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

class SimCLRTransform:
    def __init__(self, num_channels=3, is_classification=False):
        self.is_classification = is_classification
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=num_channels) if num_channels == 3 else nn.Identity(),
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __call__(self, x):
        if self.is_classification:
            return self.transform(x)  # Single view for classifier training
        return self.transform(x), self.transform(x)  # Two views for SimCLR


# ---- MMD Loss Function ----
def compute_mmd(x, y, kernel='rbf', sigma=1.0):
    """
    Computes Maximum Mean Discrepancy (MMD) loss between two sets of embeddings.

    Args:
        x: Tensor of shape (batch_size, feature_dim)
        y: Tensor of shape (batch_size, feature_dim)
        kernel: Type of kernel to use ('rbf' or 'linear')
        sigma: Bandwidth for RBF kernel

    Returns:
        MMD loss value (scalar)
    """
    def rbf_kernel(a, b, sigma):
        """Compute RBF kernel matrix"""
        norm_a = torch.sum(a**2, dim=1, keepdim=True)
        norm_b = torch.sum(b**2, dim=1, keepdim=True)
        dist_sq = norm_a + norm_b.T - 2 * torch.matmul(a, b.T)
        return torch.exp(-dist_sq / (2 * sigma**2))

    if kernel == 'rbf':
        K_xx = rbf_kernel(x, x, sigma)
        K_yy = rbf_kernel(y, y, sigma)
        K_xy = rbf_kernel(x, y, sigma)
    elif kernel == 'linear':
        K_xx = x @ x.T
        K_yy = y @ y.T
        K_xy = x @ y.T
    else:
        raise ValueError("Invalid kernel type")

    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()


# ---- NT-Xent Loss for SimCLR ----
def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]
    z = torch.cat((z_i, z_j), dim=0)  # Concatenate both views
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
        self.encoder = base_encoder(pretrained=False)
        
        # Modify first conv layer if input channels != 3
        if input_channels == 1:
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.encoder.fc = nn.Identity()  # Remove classification head
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


# ---- Training SimCLR with MMD ----
def pretrain_simclr_mmd(model, dataloader, optimizer, lambda_mmd=0.1, epochs=100, device='cuda'):
    model.train()
    for epoch in range(epochs):
        total_loss, total_nt_xent, total_mmd = 0, 0, 0
        for (x_i, x_j), _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_i, x_j = x_i.to(device), x_j.to(device)
            _, z_i = model(x_i)
            _, z_j = model(x_j)

            # Compute SimCLR loss
            nt_xent = nt_xent_loss(z_i, z_j)

            # Compute MMD loss
            mmd_loss = compute_mmd(z_i, z_j, kernel='rbf', sigma=1.0)

            # Combined loss
            loss = nt_xent + lambda_mmd * mmd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_nt_xent += nt_xent.item()
            total_mmd += mmd_loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss / len(dataloader):.4f}, "
              f"NT-Xent Loss: {total_nt_xent / len(dataloader):.4f}, MMD Loss: {total_mmd / len(dataloader):.4f}")


def get_dataset(name, is_classification=False):
    num_channels = 1 if name == 'mnist' else 3
    transform = SimCLRTransform(num_channels, is_classification)
    if name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    elif name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
    elif name == 'mnist':
        dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    else:
        raise ValueError("Unsupported dataset. Choose from 'cifar10', 'cifar100', or 'mnist'.")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['cifar10', 'cifar100', 'mnist'], help='Dataset to use')
    parser.add_argument('--lambda_mmd', type=float, default=0.1, help='Weight for MMD regularization')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = get_dataset(args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=16)
    
    input_channels = 1 if args.dataset == 'mnist' else 3
    simclr_model = SimCLR(input_channels=input_channels).to(device)
    optimizer = optim.Adam(simclr_model.parameters(), lr=1e-3, weight_decay=1e-6)
    
    print(f"Pretraining SimCLR with MMD on {args.dataset} (lambda={args.lambda_mmd})...")
    pretrain_simclr_mmd(simclr_model, train_loader, optimizer, lambda_mmd=args.lambda_mmd, epochs=5, device=device)


if __name__ == "__main__":
    main()
