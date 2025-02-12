import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import resnet18

# ---------------------
# Dataset Loader
# ---------------------
class SimCLRTransform:
    def __init__(self, num_channels=1):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)

def get_dataset():
    transform = SimCLRTransform()
    dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    return dataset

# ---------------------
# SimCLR Model
# ---------------------
class SimCLR(nn.Module):
    def __init__(self, base_encoder=resnet18, projection_dim=128, input_channels=1):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder(pretrained=False)
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

# ---------------------
# Sinkhorn Distance
# ---------------------
def sinkhorn_distance(x, y, epsilon=0.05, n_iter=10):
    batch_size = x.shape[0]
    cost_matrix = torch.cdist(x, y, p=2) ** 2 + 1e-6  # Stability
    cost_matrix = cost_matrix / cost_matrix.max().detach()
    u, v = torch.zeros(batch_size, device=x.device), torch.zeros(batch_size, device=x.device)
    
    for _ in range(n_iter):
        u_prev, v_prev = u.clone(), v.clone()
        u = -epsilon * torch.logsumexp(-cost_matrix / epsilon + v.view(1, -1), dim=1)
        v = -epsilon * torch.logsumexp(-cost_matrix / epsilon + u.view(-1, 1), dim=0)
        u -= u.mean()
        v -= v.mean()
        if torch.norm(u - u_prev) < 1e-3 and torch.norm(v - v_prev) < 1e-3:
            break

    transport_cost = torch.sum(cost_matrix * torch.exp(-cost_matrix / epsilon))
    return transport_cost / batch_size

# ---------------------
# Loss Function
# ---------------------
def contrastive_sinkhorn_loss(z_i, z_j, h_i, h_j, beta=0.1, temperature=0.5):
    batch_size = z_i.shape[0]
    z = torch.cat((z_i, z_j), dim=0)
    similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)
    
    logits = similarity_matrix / temperature
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    contrastive_loss = -torch.sum(labels * log_prob) / (2 * batch_size)
    sinkhorn_loss = sinkhorn_distance(h_i, h_j)

    return contrastive_loss + beta * sinkhorn_loss

# ---------------------
# Train SimCLR with Sinkhorn
# ---------------------
def pretrain_simclr(model, dataloader, optimizer, beta=0.1, epochs=5, device='cuda'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (x_i, x_j), _ in tqdm(dataloader, desc=f"Pretraining β={beta} Epoch {epoch+1}/{epochs}"):
            x_i, x_j = x_i.to(device), x_j.to(device)
            h_i, z_i = model(x_i)
            h_j, z_j = model(x_j)
            loss = contrastive_sinkhorn_loss(z_i, z_j, h_i, h_j, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"β={beta} Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

# ---------------------
# Train Classifier
# ---------------------
class LinearClassifier(nn.Module):
    def __init__(self, encoder):
        super(LinearClassifier, self).__init__()
        self.encoder = encoder
        self.encoder.eval()  
        for param in self.encoder.parameters():
            param.requires_grad = False  
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.fc(features)  

def train_classifier(model, dataloader, optimizer, criterion, epochs=5, device='cuda'):
    model.train()
    correct, total = 0, 0
    for epoch in range(epochs):
        for x, y in tqdm(dataloader, desc=f"Training Classifier Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total  # Classification accuracy

# ---------------------
# Experiment with Different Beta Values
# ---------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
betas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
classification_accuracies = []

train_dataset = get_dataset()
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

for beta in betas:
    model = SimCLR().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    pretrain_simclr(model, train_loader, optimizer, beta=beta, epochs=5, device=device)

    classifier = LinearClassifier(model.encoder).to(device)
    classifier_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    classifier_loader = DataLoader(classifier_dataset, batch_size=512, shuffle=True)

    classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)
    accuracy = train_classifier(classifier, classifier_loader, classifier_optimizer, nn.CrossEntropyLoss(), epochs=5, device=device)
    
    classification_accuracies.append(accuracy)
    print(f"β={beta}, Classification Accuracy: {accuracy:.4f}")

# ---------------------
# Plot Results
# ---------------------
plt.figure(figsize=(8, 5))
plt.plot(betas, classification_accuracies, marker="o", label="SinSim Performance")
plt.axhline(y=classification_accuracies[0], color='r', linestyle="--", label="SimCLR Baseline (β=0)")
plt.xlabel("Beta (Sinkhorn Regularization Strength)")
plt.ylabel("Classification Accuracy")
plt.title("Effect of Sinkhorn Regularization on SimCLR Performance")
plt.legend()
plt.grid(True)
plt.show()

##########################

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from torchvision.models import resnet18

# # ---------------------
# # Dataset Loader for CIFAR-10
# # ---------------------
# class SimCLRTransform:
#     def __init__(self, num_channels=3):
#         self.transform = transforms.Compose([
#             transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
#         ])
    
#     def __call__(self, x):
#         return self.transform(x), self.transform(x)

# def get_dataset():
#     transform = SimCLRTransform()
#     dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
#     return dataset

# # ---------------------
# # SimCLR Model for CIFAR-10
# # ---------------------
# class SimCLR(nn.Module):
#     def __init__(self, base_encoder=resnet18, projection_dim=128, input_channels=3):
#         super(SimCLR, self).__init__()
#         self.encoder = base_encoder(pretrained=False)
#         self.encoder.fc = nn.Identity()
#         self.projection_head = nn.Sequential(
#             nn.Linear(512, 512, bias=False),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, projection_dim, bias=False),
#             nn.BatchNorm1d(projection_dim, affine=False)
#         )

#     def forward(self, x):
#         h = self.encoder(x)
#         z = self.projection_head(h)
#         return h, z

# # ---------------------
# # Sinkhorn Distance
# # ---------------------
# def sinkhorn_distance(x, y, epsilon=0.05, n_iter=10):
#     batch_size = x.shape[0]
#     cost_matrix = torch.cdist(x, y, p=2) ** 2 + 1e-6  # Stability
#     cost_matrix = cost_matrix / cost_matrix.max().detach()
#     u, v = torch.zeros(batch_size, device=x.device), torch.zeros(batch_size, device=x.device)
    
#     for _ in range(n_iter):
#         u_prev, v_prev = u.clone(), v.clone()
#         u = -epsilon * torch.logsumexp(-cost_matrix / epsilon + v.view(1, -1), dim=1)
#         v = -epsilon * torch.logsumexp(-cost_matrix / epsilon + u.view(-1, 1), dim=0)
#         u -= u.mean()
#         v -= v.mean()
#         if torch.norm(u - u_prev) < 1e-3 and torch.norm(v - v_prev) < 1e-3:
#             break

#     transport_cost = torch.sum(cost_matrix * torch.exp(-cost_matrix / epsilon))
#     return transport_cost / batch_size

# # ---------------------
# # Loss Function
# # ---------------------
# def contrastive_sinkhorn_loss(z_i, z_j, h_i, h_j, beta=0.1, temperature=0.5):
#     batch_size = z_i.shape[0]
#     z = torch.cat((z_i, z_j), dim=0)
#     similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
#     labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
#     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)
    
#     logits = similarity_matrix / temperature
#     logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()
#     exp_logits = torch.exp(logits)
#     log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

#     contrastive_loss = -torch.sum(labels * log_prob) / (2 * batch_size)
#     sinkhorn_loss = sinkhorn_distance(h_i, h_j)

#     return contrastive_loss + beta * sinkhorn_loss

# # ---------------------
# # Train SimCLR with Sinkhorn
# # ---------------------
# def pretrain_simclr(model, dataloader, optimizer, beta=0.1, epochs=5, device='cuda'):
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for (x_i, x_j), _ in tqdm(dataloader, desc=f"Pretraining β={beta} Epoch {epoch+1}/{epochs}"):
#             x_i, x_j = x_i.to(device), x_j.to(device)
#             h_i, z_i = model(x_i)
#             h_j, z_j = model(x_j)
#             loss = contrastive_sinkhorn_loss(z_i, z_j, h_i, h_j, beta=beta)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"β={beta} Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

# # ---------------------
# # Train Classifier
# # ---------------------
# class LinearClassifier(nn.Module):
#     def __init__(self, encoder):
#         super(LinearClassifier, self).__init__()
#         self.encoder = encoder
#         self.encoder.eval()  
#         for param in self.encoder.parameters():
#             param.requires_grad = False  
#         self.fc = nn.Linear(512, 10)

#     def forward(self, x):
#         with torch.no_grad():
#             features = self.encoder(x)
#         return self.fc(features)  

# def train_classifier(model, dataloader, optimizer, criterion, epochs=10, device='cuda'):
#     model.train()
#     correct, total = 0, 0
#     for epoch in range(epochs):
#         for x, y in tqdm(dataloader, desc=f"Training Classifier Epoch {epoch+1}/{epochs}"):
#             x, y = x.to(device), y.to(device)
#             logits = model(x)
#             loss = criterion(logits, y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             preds = torch.argmax(logits, dim=1)
#             correct += (preds == y).sum().item()
#             total += y.size(0)
#     return correct / total  # Classification accuracy

# # ---------------------
# # Experiment with Different Beta Values
# # ---------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# betas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# classification_accuracies = []

# train_dataset = get_dataset()
# train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# for beta in betas:
#     model = SimCLR().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
#     pretrain_simclr(model, train_loader, optimizer, beta=beta, epochs=5, device=device)

#     classifier = LinearClassifier(model.encoder).to(device)
#     classifier_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
#     classifier_loader = DataLoader(classifier_dataset, batch_size=512, shuffle=True)

#     classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)
#     accuracy = train_classifier(classifier, classifier_loader, classifier_optimizer, nn.CrossEntropyLoss(), epochs=5, device=device)
    
#     classification_accuracies.append(accuracy)
#     print(f"β={beta}, Classification Accuracy: {accuracy:.4f}")

# # ---------------------
# # Plot Results
# # ---------------------
# plt.figure(figsize=(8, 5))
# plt.plot(betas, classification_accuracies, marker="o", label="SinSim Performance")
# plt.axhline(y=classification_accuracies[0], color='r', linestyle="--", label="SimCLR Baseline (β=0)")
# plt.xlabel("Beta (Sinkhorn Regularization Strength)")
# plt.ylabel("Classification Accuracy")
# plt.title("Effect of Sinkhorn Regularization on SimCLR Performance (CIFAR-10)")
# plt.legend()
# plt.grid(True)
# plt.show()



#############


# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torchvision.models import resnet18
# from tqdm import tqdm

# # ---------------------
# # Load CIFAR-10 Dataset
# # ---------------------
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
# ])

# batch_size = 512
# cifar10_train = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
# train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)

# # ---------------------
# # Define SimCLR Model (Feature Extractor)
# # ---------------------
# class SimCLR(nn.Module):
#     def __init__(self):
#         super(SimCLR, self).__init__()
#         self.encoder = resnet18(pretrained=False)
#         self.encoder.fc = nn.Identity()  # Remove classification head

#     def forward(self, x):
#         h = self.encoder(x)  # Extract intermediate feature representations
#         return h

# # ---------------------
# # Compute Variance of Feature Representations
# # ---------------------
# def compute_feature_variance(model, dataloader, device="cuda"):
#     model.to(device)
#     model.eval()
    
#     all_features = []
#     with torch.no_grad():
#         for images, _ in tqdm(dataloader, desc="Extracting Features"):
#             images = images.to(device)
#             features = model(images)  # Extract intermediate features
#             all_features.append(features.cpu().numpy())
    
#     all_features = np.concatenate(all_features, axis=0)  # Shape: (N, feature_dim)
#     feature_variance = np.var(all_features, axis=0)  # Variance along feature dimension
#     return np.mean(feature_variance)  # Mean variance across all features

# # ---------------------
# # Run Experiment for SimCLR (β=0) and SinSim (β=0.5)
# # ---------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Run SimCLR (β=0)
# model_simclr = SimCLR().to(device)
# variance_simclr = compute_feature_variance(model_simclr, train_loader, device)

# # Run SinSim (β=0.5)
# model_sinsim = SimCLR().to(device)  # Assume SinSim uses the same model structure
# variance_sinsim = compute_feature_variance(model_sinsim, train_loader, device)

# # Print Results
# print("\nFeature Variance Results:")
# print(f"SimCLR (β=0)  : {variance_simclr:.6f}")
# print(f"SinSim (β=0.5): {variance_sinsim:.6f}")

# # Interpretation:
# # - If SinSim has a higher variance than SimCLR, it suggests better feature dispersion.
# # - If variance is lower, the representations may still be collapsing.
