import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from vision_transformer import ViT


def init_dataset(dataset_name, batch_size=128, samples_per_class=10):
    num_classes = 0
    assert dataset_name in ['mnist', 'cifar10', 'cifar100'], "must be one of mnist, cifar10, cifar100"

    if dataset_name == 'mnist':
        train_data, test_data = mnist_init_dataset()
        num_classes = 10
        subset_targets = train_data.targets.numpy()
    elif dataset_name == 'cifar10':
        train_data, test_data = cifar10_init_dataset()
        num_classes = 10
        subset_targets = np.array(train_data.targets)
    elif dataset_name == 'cifar100':
        train_data, test_data = cifar100_init_dataset()
        num_classes = 100
        subset_targets = np.array(train_data.targets)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Collect indices (balanced subset)
    selected_indices = []
    for class_id in range(num_classes):
        class_indices = np.where(subset_targets == class_id)[0]
        chosen = np.random.choice(class_indices, size=samples_per_class, replace=False)
        selected_indices.extend(chosen)

    # Create a subset DataLoader (shuffling not needed)
    subset = Subset(train_data, selected_indices)
    subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    # Print the number of samples in each dataset
    print(f"{len(train_data)} samples in train data")
    print(f"{len(test_data)} samples in test data")
    print(f"{len(subset)} samples in visualization subset")

    return train_loader, test_loader, subset_loader

def mnist_init_dataset():
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return train_data, test_data


def cifar10_init_dataset():
    # The transform pipeline applies two operations: conversion to a tensor and normalization using CIFAR-10
    # dataset-specific mean and standard deviation values.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # Load CIFAR-10 dataset
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return train_data, test_data

def cifar100_init_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    return train_data, test_data


def init_mlp_for_dataset(dataset_name, hidden_dims=[128, 64], dropout=0.2):
    if dataset_name == "mnist":
        input_size = 28 * 28  # MNIST image size
        num_classes = 10
    elif dataset_name == "cifar10":
        input_size = 32 * 32 * 3  # CIFAR image size (RGB channels)
        num_classes = 10
    elif dataset_name == "cifar100":
        input_size = 64 * 64 * 3 # CIFAR-100 image size (RGB channels)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return MLP(hidden_dims=hidden_dims, input_size=input_size, num_classes=num_classes, dropout=dropout)

class MLP(nn.Module):
    def __init__(self, hidden_dims=[128, 64], input_size=784, num_classes=10, dropout=0.2):
        super(MLP, self).__init__()
        self.emb_dim = hidden_dims[-1]

        layers = []
        prev_dim = input_size
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))  # Add dropout here
            prev_dim = dim

        self.feature_extractor = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, num_classes)

    def forward(self, x, return_embedding=False):
        x = x.view(x.size(0), -1)
        h = self.feature_extractor(x)
        out = self.head(h)
        if return_embedding:
            return out, h
        return out


def init_cnn_for_dataset(dataset_name, hidden_dim=128):
    if dataset_name == "mnist":
        input_channels = 1
        num_classes = 10
    elif dataset_name == "cifar10":
        input_channels = 3
        num_classes = 10
    elif dataset_name == "cifar100":
        input_channels = 3
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return CNN(hidden_dim=hidden_dim, num_classes=num_classes, input_channels=input_channels)

class CNN(nn.Module):
    def __init__(self, hidden_dim=128, conv1_out_channels=32, conv2_out_channels=64, num_classes=10, input_channels=1):
        super(CNN, self).__init__()
        self.emb_dim = hidden_dim
        self.conv1 = nn.Conv2d(input_channels, conv1_out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the correct input size for fc1
        if input_channels == 1:
            self.fc1_input_dim = conv2_out_channels * 7 * 7  # MNIST size after pooling
        else:
            self.fc1_input_dim = conv2_out_channels * 8 * 8  # CIFAR size after pooling

        self.fc1 = nn.Linear(self.fc1_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        if return_embedding:
            return out, h
        return out

def init_vit_for_dataset(dataset_name, emb_dim=128, depth=6, num_heads=4, mlp_dim=256, dropout=0.1):
    if dataset_name == "mnist":
        img_size = 28
        num_classes = 10
        input_channels = 1  # MNIST grayscale (would require adjustment)
    elif dataset_name == "cifar10":
        img_size = 32
        num_classes = 10
        input_channels = 3
    elif dataset_name == "cifar100":
        img_size = 32
        num_classes = 100
        input_channels = 3
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if input_channels != 3:
        raise ValueError("Current ViT implementation supports only RGB images (3 channels).")

    model = ViT(
        img_size=img_size,
        patch_size=4,
        num_classes=num_classes,
        emb_dim=emb_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
    )
    return model