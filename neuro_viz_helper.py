import torch
from torch.utils.data import DataLoader
from NeuroVisualizer.neuro_aux.trajectories_data import NormalizeModelParameters
from NeuroVisualizer.neuro_aux.utils import repopulate_model
from tqdm import tqdm

### Dataset and Loader
class FlatTensorDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        x = torch.load(self.file_paths[idx], map_location='cpu')
        if self.transform:
            x = self.transform(x)
        return x

def calculate_mean_std_flat(file_paths):
    tensors = [torch.load(fp, map_location='cpu') for fp in file_paths]
    stacked = torch.stack(tensors)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    return mean, std

def get_dataloader_flat(pt_files, batch_size, shuffle=True, num_workers=2):
    mean, std = calculate_mean_std_flat(pt_files)
    normalizer = NormalizeModelParameters(mean, std)
    dataset = FlatTensorDataset(pt_files, transform=normalizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), normalizer

### Grid Generation
def generate_latent_grid(min_map=-1, max_map=1, xnum=25, device='cpu'):
    step_size = (max_map - min_map) / xnum
    x_coords = torch.arange(min_map, max_map + step_size, step_size)
    y_coords = torch.arange(min_map, max_map + step_size, step_size)
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing="ij")
    grid_coords = torch.stack((xx.flatten(), yy.flatten()), dim=1).to(device)
    return xx, yy, grid_coords

### Loss Computation
def compute_grid_losses(grid_coords, transform, ae_model, model, loss_obj, loss_name, whichloss, device):
    rec_grid_models = ae_model.decoder(grid_coords)
    rec_grid_models = rec_grid_models * transform.std.to(device) + transform.mean.to(device)

    grid_losses = []

    for i in tqdm(range(rec_grid_models.shape[0]), desc="Computing grid losses"):
        model_flattened = rec_grid_models[i, :]
        model_repopulated = repopulate_model(
            model_flattened,
            model
        )
        model_repopulated.eval()
        model_repopulated = model_repopulated.to(device)
        loss = loss_obj.get_loss(model_repopulated, loss_name, whichloss).detach()
        grid_losses.append(loss)

    grid_losses = torch.stack(grid_losses)
    return grid_losses


from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch



# Import your dataset init helpers from vision_classification
from vision_classification import (
    mnist_init_dataset,
    cifar10_init_dataset,
    cifar100_init_dataset
)

class Loss:
    def __init__(self, dataset_name, device):
        self.device = device

        if dataset_name.lower() == "mnist":
            train_dataset, test_dataset = mnist_init_dataset()

        elif dataset_name.lower() == "cifar10":
            train_dataset, _, test_dataset = cifar10_init_dataset()

        elif dataset_name.lower() == "cifar100":
            train_dataset, _, test_dataset = cifar100_init_dataset()

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        kwargs = {'num_workers': 2, 'pin_memory': True}
        self.train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, **kwargs)

    def get_loss(self, dnn, loss_name, whichloss):
        if whichloss == "mse":
            loss = 0
            loader = self.test_loader if loss_name == "test_loss" else self.train_loader
            with torch.no_grad():
                for data, target in loader:
                    output = dnn(data.to(self.device))
                    loss += F.nll_loss(output, target.to(self.device), reduction='sum').item()
            loss /= len(loader.dataset)
            return torch.tensor(loss)
        else:
            raise ValueError(f"Loss type not defined: {whichloss}")

def compute_trajectory(
        trajectory_loader,
        ae_model,
        transform,
        loss_obj,
        model,
        loss_name,
        whichloss,
        device,
):
    """
    Computes:
    - trajectory_coordinates (2D latent space)
    - trajectory_models (decoded weights)
    - trajectory_losses (loss values)
    """
    # ---- 2️⃣ Decode trajectory ----
    trajectory_models = []
    trajectory_coordinates = []

    ae_model.eval()
    with torch.no_grad():
        for batch in tqdm(trajectory_loader, desc="Decoding trajectory"):
            batch = batch.to(device)
            x_recon, z = ae_model(batch)
            trajectory_coordinates.append(z.cpu())
            x_recon = x_recon * transform.std.to(device) + transform.mean.to(device)
            trajectory_models.append(x_recon.cpu())

    trajectory_coordinates = torch.cat(trajectory_coordinates, dim=0)
    trajectory_models = torch.cat(trajectory_models, dim=0)

    print(f"✅ Decoded trajectory shapes: coords {trajectory_coordinates.shape}, models {trajectory_models.shape}")

    # ---- 3️⃣ Compute losses ----
    from vision_classification import init_mlp_for_dataset

    trajectory_losses = []
    for i in tqdm(range(trajectory_models.shape[0]), desc="Computing trajectory losses"):
        model_flattened = trajectory_models[i, :]
        repopulate_model(model_flattened, model)
        model.eval()
        loss = loss_obj.get_loss(model, loss_name, whichloss).detach()
        trajectory_losses.append(loss)

    trajectory_losses = torch.stack(trajectory_losses)

    print(f"✅ Computed {trajectory_losses.shape[0]} trajectory losses")

    return trajectory_coordinates, trajectory_models, trajectory_losses