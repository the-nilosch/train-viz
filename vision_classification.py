from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np

from models.cnn import CNN
from models.dense_net import DenseNet
from models.mlp import MLP
from models.res_net import ResNet
from models.vision_transformer import ViT


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


def init_densenet_for_dataset(
    dataset_name,
    block_config=(6, 12, 24),  # ähnlich DenseNet-121
    growth_rate=16,
    fc_hidden_dims=[128],
    dropout=0.1
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

    model = DenseNet(
        block_config=block_config,
        growth_rate=growth_rate,
        num_classes=num_classes,
        input_channels=input_channels,
        fc_hidden_dims=fc_hidden_dims,
        dropout=dropout
    )
    return model