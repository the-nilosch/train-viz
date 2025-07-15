import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import os
from torch.utils.data import DataLoader
from NeuroVisualizer.neuro_aux.trajectories_data import NormalizeModelParameters
from NeuroVisualizer.neuro_aux.utils import repopulate_model
from tqdm import tqdm

from helper.vision_classification import (
    mnist_init_dataset,
    cifar10_init_dataset,
    cifar100_init_dataset
)

### Dataset and Loader
class FlatTensorDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        x = torch.load(self.file_paths[idx], map_location='cpu', weights_only=True)
        if self.transform:
            x = self.transform(x)
        return x

def calculate_mean_std_flat(file_paths):
    tensors = [torch.load(fp, map_location='cpu', weights_only=True) for fp in file_paths]
    stacked = torch.stack(tensors)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    return mean, std

def get_dataloader_flat(pt_files, batch_size, shuffle=True, num_workers=2):
    mean, std = calculate_mean_std_flat(pt_files)
    normalizer = NormalizeModelParameters(mean, std)
    dataset = FlatTensorDataset(pt_files, transform=normalizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), normalizer

### Training

def train_autoencoder(
    model,
    train_loader,
    device,
    save_path='best_ae_model.pt',
    num_epochs=100,
    lr=1e-3,
    patience=10,
    log_every=5,
    verbose=True
):
    """
    Generic improved AE training loop with early stopping and scheduler.
    Saves best model to save_path.
    """
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    epochs_no_improve = 0

    losses_log = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        losses_log.append({'epoch': epoch, 'loss': avg_loss})

        if verbose:
            print(f"Epoch {epoch:03d} - Avg Loss: {avg_loss:.6f}")

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            if verbose:
                print(f"✅ New best model saved with loss {best_loss:.6f}")
        else:
            epochs_no_improve += 1
            if verbose:
                print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

        # Log to CSV every few epochs
        if epoch % log_every == 0:
            df_log = pd.DataFrame(losses_log)
            df_log.to_csv(os.path.join(os.path.dirname(save_path), 'ae_training_log.csv'), index=False)

    # Final save
    df_log = pd.DataFrame(losses_log)
    df_log.to_csv(os.path.join(os.path.dirname(save_path), 'ae_training_log.csv'), index=False)
    if verbose:
        print(f"Training complete. Best loss: {best_loss:.6f}")

    return model



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
        self.train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, **kwargs)

    def get_loss(self, dnn, loss_name, whichloss):
        if whichloss == "mse" or whichloss == "crossentropy":
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
    # ---- Decode trajectory ----
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

    # ---- Compute losses ----

    trajectory_losses = []
    for i in tqdm(range(trajectory_models.shape[0]), desc="Computing trajectory losses"):
        model_flattened = trajectory_models[i, :]
        assert model_flattened.numel() == sum(p.numel() for p in model.parameters()); "Mismatch in parameter size"
        repopulate_model_fixed(model_flattened, model)
        model.eval()
        loss = loss_obj.get_loss(model, loss_name, whichloss).detach()
        trajectory_losses.append(loss)

    trajectory_losses = torch.stack(trajectory_losses)

    print(f"✅ Computed {trajectory_losses.shape[0]} trajectory losses")

    return trajectory_coordinates, trajectory_models, trajectory_losses

def repopulate_model_fixed(flattened_params, model):
    start_idx = 0
    for param in model.parameters():
        size = param.numel()
        sub_flattened = flattened_params[start_idx : start_idx + size].view(param.size())
        param.data.copy_(sub_flattened)
        start_idx += size
    return model