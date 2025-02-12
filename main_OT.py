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

experiment = "CIfar_300_OT"

def reset_batchnorm_running_stats(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()   

class SimCLRTransform:
    def __init__(self, num_channels=3, is_classification=False):
        self.is_classification = is_classification
        if is_classification:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])  
        else:
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
        if self.is_classification:
            return self.transform(x)   
        return self.transform(x), self.transform(x)   

# ---- Dataset Loader ----
def get_dataset(name, train=True, is_classification=False):
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

def train_classifier(model, dataloader, optimizer, criterion, epochs=20, device='cuda'):

    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)

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


def save_encoder(model, path=f"simclr_encoder_{experiment}.pth"):
    torch.save(model.encoder.state_dict(), path)
    print(f"SimCLR encoder saved to {path}")

def load_encoder(model, path=f"simclr_encoder_{experiment}.pth"):
    model.encoder.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.encoder.eval()
    print(f"SimCLR encoder loaded from {path}")
   
   
def sinkhorn_distance(x, y, epsilon=0.05, n_iter=50):

    batch_size = x.shape[0]
    
    cost_matrix = torch.cdist(x, y, p=2) ** 2 + 1e-6  
    cost_matrix = cost_matrix / cost_matrix.max().detach()

    


    u = torch.zeros(batch_size, device=x.device)
    v = torch.zeros(batch_size, device=x.device)

    for _ in range(n_iter):
        u_prev, v_prev = u.clone(), v.clone() 
        u = -epsilon * torch.logsumexp(-cost_matrix / epsilon + v.view(1, -1), dim=1)
        v = -epsilon * torch.logsumexp(-cost_matrix / epsilon + u.view(-1, 1), dim=0)


        # import pdb;pdb.set_trace()

        u -= u.mean()
        v -= v.mean()

        if torch.norm(u - u_prev) < 1e-3 and torch.norm(v - v_prev) < 1e-3:
            break

    transport_cost = torch.sum(cost_matrix * torch.exp(-cost_matrix / epsilon))
    
    return transport_cost / batch_size


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



def contrastive_sinkhorn_loss(z_i, z_j, h_i, h_j, temperature=0.5, lambda_sinkhorn=0.8):

    contrastive_loss = nt_xent_loss(z_i, z_j, temperature=temperature)

    sinkhorn_reg = sinkhorn_distance(h_i, h_j)

    # import pdb; pdb.set_trace()
    total_loss = contrastive_loss + lambda_sinkhorn * sinkhorn_reg

    return total_loss


class SimCLR(nn.Module):
    def __init__(self, base_encoder=resnet18, projection_dim=128, input_channels=3):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder(pretrained=False)

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

class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.encoder = encoder
        self.encoder.eval()  
        for param in self.encoder.parameters():
            param.requires_grad = False  
        self.fc = nn.Linear(512, num_classes)  

    def forward(self, x):
        with torch.no_grad():  
            features = self.encoder(x)
        return self.fc(features)  

# ---- Pretrain SimCLR with OT Loss ----
def pretrain_simclr(model, dataloader, optimizer, epochs=5, device='cuda'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (x_i, x_j), _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_i, x_j = x_i.to(device), x_j.to(device)
            h_i, z_i = model(x_i)
            h_j, z_j = model(x_j)

            loss = contrastive_sinkhorn_loss(z_i, z_j, h_i, h_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

    save_encoder(model)

# ---- Main Function ----
def main():
    dataset_name = "cifar10"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = get_dataset(dataset_name, train=True)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    model = SimCLR(input_channels=1 if dataset_name == "mnist" else 3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    try:
        load_encoder(model)
    except:
        pretrain_simclr(model, train_loader, optimizer, epochs=300, device=device)

    classifier = LinearClassifier(model.encoder).to(device)
    classifier_dataset = get_dataset(dataset_name, train=True, is_classification=True)
    classifier_dataset_loader = DataLoader(classifier_dataset, batch_size=512, shuffle=True)

    reset_batchnorm_running_stats(classifier.encoder)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)
    train_classifier(classifier, classifier_dataset_loader, classifier_optimizer, nn.CrossEntropyLoss(), epochs=10, device=device)

if __name__ == "__main__":
    main()
