import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from res_net import ResNet
from vision_transformer import ViT


def init_dataset(dataset_name, batch_size=128, samples_per_class=10):
    num_classes = 0
    assert dataset_name in ['mnist', 'cifar10', 'cifar100'], "must be one of mnist, cifar10, cifar100"

    if dataset_name == 'mnist':
        train_data, test_data = mnist_init_dataset()
        eval_data = train_data
        num_classes = 10
        subset_targets = train_data.targets.numpy()
    elif dataset_name == 'cifar10':
        train_data, eval_data, test_data = cifar10_init_dataset()
        num_classes = 10
        subset_targets = np.array(eval_data.targets)
    elif dataset_name == 'cifar100':
        train_data, eval_data, test_data = cifar100_init_dataset()
        subset_targets = np.array(eval_data.targets)
        num_classes = 100

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Collect indices (balanced subset)
    selected_indices = []
    for class_id in range(num_classes):
        class_indices = np.where(subset_targets == class_id)[0]
        chosen = np.random.choice(class_indices, size=samples_per_class, replace=False)
        selected_indices.extend(chosen)

    # Create a subset DataLoader (shuffling not needed)
    subset = Subset(eval_data, selected_indices)
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
    # Training transform (with augmentation)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # Subset and Test transform (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # Load CIFAR-10 dataset
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    eval_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=eval_transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_transform)

    return train_data, eval_data, test_data

def cifar100_init_dataset():
    # Training transform (with augmentation)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    eval_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=eval_transform)
    test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=eval_transform)

    return train_data, eval_data, test_data

def get_text_labels(dataset_name):
    if dataset_name == "mnist":
        return [str(i) for i in range(10)]
    elif dataset_name == "cifar10":
        return datasets.CIFAR10(root='./data', download=True).classes
    elif dataset_name == "cifar100":
        return datasets.CIFAR100(root='./data', download=True).classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_cifar100_coarse_to_fine_labels():
    return {
        "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
        "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
        "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
        "food containers": ["bottle", "bowl", "can", "cup", "plate"],
        "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
        "household electrical devices": ["clock", "keyboard", "lamp", "telephone", "television"],
        "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
        "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
        "largecarnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
        "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
        "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
        "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
        "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
        "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
        "people": ["baby", "boy", "girl", "man", "woman"],
        "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
        "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
        "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
        "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
        "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
    }

def get_cifar100_fine_to_coarse_labels():
    coarse_to_fine = get_cifar100_coarse_to_fine_labels()
    return {
        fine: coarse
        for coarse, fine_list in coarse_to_fine.items()
        for fine in fine_list
    }

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

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class MLP(nn.Module):
    def __init__(self, hidden_dims=[128, 64], input_size=784, num_classes=10, dropout=0.2):
        super(MLP, self).__init__()
        self.emb_dim = hidden_dims[-1]

        layers = []
        prev_dim = input_size
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim

        self.feature_extractor = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, num_classes)

        # Apply Kaiming Initialization
        self.apply(init_weights_kaiming)

    def forward(self, x, return_embedding=False):
        x = x.view(x.size(0), -1)
        h = self.feature_extractor(x)
        out = self.head(h)
        if return_embedding:
            return out, h
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Ensure residual connection can handle dimension change
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        # Apply residual connection
        if self.residual_conv:
            identity = self.residual_conv(identity)

        out += identity
        out = self.relu(out)
        return out


class CNN(nn.Module):
    def __init__(self, conv_dims=[64, 128], kernel_sizes=[3, 3], hidden_dims=[128], num_classes=10, input_channels=3, dropout=0.2):
        super(CNN, self).__init__()
        self.emb_dim = hidden_dims[-1]

        # Validate kernel size length
        if len(kernel_sizes) != len(conv_dims):
            raise ValueError(f"Expected kernel_sizes of length {len(conv_dims)}, got {len(kernel_sizes)}")

        # Construct residual blocks
        conv_layers = []
        prev_channels = input_channels

        for out_channels, kernel_size in zip(conv_dims, kernel_sizes):
            conv_layers.append(ResidualBlock(prev_channels, out_channels, kernel_size, stride=1, dropout=dropout))
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate input size for fully connected layers
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, 32, 32)
            dummy_output = self.conv_layers(dummy_input)
            fc_input_dim = dummy_output.view(1, -1).shape[1]

        # Construct fully connected layers
        fc_layers = []
        prev_dim = fc_input_dim

        for dim in hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = dim

        self.fc_layers = nn.Sequential(*fc_layers)
        self.head = nn.Linear(prev_dim, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        h = self.fc_layers(x)
        out = self.head(h)

        if return_embedding:
            return out, h
        return out

def init_cnn_for_dataset(dataset_name, conv_dims=[64, 128], kernel_sizes=[3, 3], hidden_dims=[128], dropout=0.2):
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

    return CNN(conv_dims=conv_dims, kernel_sizes=kernel_sizes, hidden_dims=hidden_dims, num_classes=num_classes, input_channels=input_channels, dropout=dropout)

def init_vit_for_dataset(dataset_name, emb_dim=128, depth=6, num_heads=4, mlp_dim=256, dropout=0.1, patch_size=4):
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


    model = ViT(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        emb_dim=emb_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        input_channels=input_channels
    )
    return model

def init_resnet_for_dataset(
    dataset_name,
    layers=[2, 2, 2, 2],
    fc_hidden_dims=[],
    dropout=0.2,
    zero_init_residual=False
):
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

    model = ResNet(
        layers=layers,
        num_classes=num_classes,
        input_channels=input_channels,
        fc_hidden_dims=fc_hidden_dims,
        dropout=dropout,
        zero_init_residual=zero_init_residual
    )
    return model
