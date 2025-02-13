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

# class SimCLRTransform:
#     def __init__(self, num_channels=1):
#         self.transform = transforms.Compose([
#             transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
    
#     def __call__(self, x):
#         return self.transform(x), self.transform(x)

# def get_dataset():
#     transform = SimCLRTransform()
#     dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
#     return dataset


# class SimCLR(nn.Module):
#     def __init__(self, base_encoder=resnet18, projection_dim=128, input_channels=1):
#         super(SimCLR, self).__init__()
#         self.encoder = base_encoder(pretrained=False)
#         self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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

# def train_classifier(model, dataloader, optimizer, criterion, epochs=5, device='cuda'):
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
#     classifier_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
#     classifier_loader = DataLoader(classifier_dataset, batch_size=512, shuffle=True)

#     classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)
#     accuracy = train_classifier(classifier, classifier_loader, classifier_optimizer, nn.CrossEntropyLoss(), epochs=5, device=device)
    
#     classification_accuracies.append(accuracy)
#     print(f"β={beta}, Classification Accuracy: {accuracy:.4f}")


# plt.figure(figsize=(8, 5))
# plt.plot(betas, classification_accuracies, marker="o", label="SinSim Performance")
# plt.axhline(y=classification_accuracies[0], color='r', linestyle="--", label="SimCLR Baseline (β=0)")
# plt.xlabel("Beta (Sinkhorn Regularization Strength)")
# plt.ylabel("Classification Accuracy")
# plt.title("Effect of Sinkhorn Regularization on SimCLR Performance")
# plt.legend()
# plt.grid(True)
# plt.show()

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


# plt.figure(figsize=(8, 5))
# plt.plot(betas, classification_accuracies, marker="o", label="SinSim Performance")
# plt.axhline(y=classification_accuracies[0], color='r', linestyle="--", label="SimCLR Baseline (β=0)")
# plt.xlabel("Beta (Sinkhorn Regularization Strength)")
# plt.ylabel("Classification Accuracy")
# plt.title("Effect of Sinkhorn Regularization on SimCLR Performance (CIFAR-10)")
# plt.legend()
# plt.grid(True)
# plt.show()



#######################



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

# class SimCLRTransform:
#     def __init__(self, num_channels=1):
#         self.transform = transforms.Compose([
#             transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
    
#     def __call__(self, x):
#         return self.transform(x), self.transform(x)

# def get_dataset():
#     transform = SimCLRTransform()
#     dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
#     return dataset


# class SimCLR(nn.Module):
#     def __init__(self, base_encoder=resnet18, projection_dim=128, input_channels=1):
#         super(SimCLR, self).__init__()
#         self.encoder = base_encoder(pretrained=False)
#         self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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


# def contrastive_sinkhorn_loss(z_i, z_j, h_i, h_j, epsilon=0.05, beta=0.1, temperature=0.5):
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
#     sinkhorn_loss = sinkhorn_distance(h_i, h_j, epsilon=epsilon)

#     return contrastive_loss + beta * sinkhorn_loss

# def pretrain_simclr(model, dataloader, optimizer, epsilon=0.05, beta=0.1, epochs=5, device='cuda'):
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for (x_i, x_j), _ in tqdm(dataloader, desc=f"Pretraining ε={epsilon} Epoch {epoch+1}/{epochs}"):
#             x_i, x_j = x_i.to(device), x_j.to(device)
#             h_i, z_i = model(x_i)
#             h_j, z_j = model(x_j)
#             loss = contrastive_sinkhorn_loss(z_i, z_j, h_i, h_j, epsilon=epsilon, beta=beta)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"ε={epsilon} Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")


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

# def train_classifier(model, dataloader, optimizer, criterion, epochs=5, device='cuda'):
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


# device = "cuda" if torch.cuda.is_available() else "cpu"
# epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ]
# classification_accuracies = []

# train_dataset = get_dataset()
# train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# for epsilon in epsilons:
#     model = SimCLR().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
#     pretrain_simclr(model, train_loader, optimizer, epsilon=epsilon, beta=0.1, epochs=5, device=device)

#     classifier = LinearClassifier(model.encoder).to(device)
#     classifier_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
#     classifier_loader = DataLoader(classifier_dataset, batch_size=512, shuffle=True)

#     classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)
#     accuracy = train_classifier(classifier, classifier_loader, classifier_optimizer, nn.CrossEntropyLoss(), epochs=5, device=device)
    
#     classification_accuracies.append(accuracy)
#     print(f"ε={epsilon}, Classification Accuracy: {accuracy:.4f}")


# plt.figure(figsize=(8, 5))
# plt.plot(epsilons, classification_accuracies, marker="o", label="SinSim Performance")
# plt.axhline(y=classification_accuracies[0], color='r', linestyle="--", label="Baseline ε=0.01")
# plt.xlabel("Epsilon (Sinkhorn Entropy Regularization Strength)")
# plt.ylabel("Classification Accuracy")
# plt.title("Effect of Epsilon on Sinkhorn Regularized SimCLR Performance")
# plt.legend()
# plt.grid(True)
# plt.show()

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


def contrastive_sinkhorn_loss(z_i, z_j, h_i, h_j, n_iter=10, beta=0.1, epsilon=0.05, temperature=0.5):
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
    sinkhorn_loss = sinkhorn_distance(h_i, h_j, epsilon=epsilon, n_iter=n_iter)

    return contrastive_loss + beta * sinkhorn_loss

def pretrain_simclr(model, dataloader, optimizer, n_iter=10, beta=0.1, epsilon=0.05, epochs=5, device='cuda'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (x_i, x_j), _ in tqdm(dataloader, desc=f"Pretraining n_iter={n_iter} Epoch {epoch+1}/{epochs}"):
            x_i, x_j = x_i.to(device), x_j.to(device)
            h_i, z_i = model(x_i)
            h_j, z_j = model(x_j)
            loss = contrastive_sinkhorn_loss(z_i, z_j, h_i, h_j, n_iter=n_iter, beta=beta, epsilon=epsilon)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"n_iter={n_iter} Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")


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


device = "cuda" if torch.cuda.is_available() else "cpu"
iterations = [10, 20, 30, 40, 50]
classification_accuracies = []

train_dataset = get_dataset()
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

for n_iter in iterations:
    model = SimCLR().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    pretrain_simclr(model, train_loader, optimizer, n_iter=n_iter, beta=0.1, epsilon=0.05, epochs=5, device=device)

    classifier = LinearClassifier(model.encoder).to(device)
    classifier_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    classifier_loader = DataLoader(classifier_dataset, batch_size=512, shuffle=True)

    classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)
    accuracy = train_classifier(classifier, classifier_loader, classifier_optimizer, nn.CrossEntropyLoss(), epochs=5, device=device)
    
    classification_accuracies.append(accuracy)
    print(f"n_iter={n_iter}, Classification Accuracy: {accuracy:.4f}")


plt.figure(figsize=(8, 5))
plt.plot(iterations, classification_accuracies, marker="o", label="SinSim Performance")
plt.axhline(y=classification_accuracies[0], color='r', linestyle="--", label="Baseline n_iter=10")
plt.xlabel("Number of Sinkhorn Iterations")
plt.ylabel("Classification Accuracy")
plt.title("Effect of Sinkhorn Iterations on SimCLR Performance")
plt.legend()
plt.grid(True)
plt.show()


