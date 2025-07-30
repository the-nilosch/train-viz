from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
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

def calculate_mean_std_flat(pt_files):
    all_tensors = []
    for file in pt_files:
        tensor = torch.load(file, map_location='cpu', weights_only=True)
        all_tensors.append(tensor)
    stacked = torch.stack(all_tensors)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    return mean, std

def get_dataloader_flat(
    pt_files_per_run: list[list[str]],
    batch_size,
    shuffle=False, num_workers=2,
    oversample_later=False, power: float = 1.0,
    include_lmc=False,
    num_lmc_points=10
):
    if include_lmc:
        pt_files_per_run = add_lmc_paths(pt_files_per_run, num_lmc_points)

    print(len(pt_files_per_run))
    pt_files = [fp for run in pt_files_per_run for fp in run]

    # compute global mean/std as before
    mean, std = calculate_mean_std_flat(pt_files)
    normalizer = NormalizeModelParameters(mean, std)
    dataset = FlatTensorDataset(pt_files, transform=normalizer)

    if oversample_later:
        N = len(dataset)
        # weight_i ∝ ((i+1)/N)**power — higher power ⇒ more focus on latest
        raw = [(i+1)/N for i in range(N)]
        weights = [r**power for r in raw]
        sampler = WeightedRandomSampler(weights, num_samples=N, replacement=True)
        return DataLoader(dataset, batch_size=batch_size,
                          sampler=sampler, num_workers=num_workers), normalizer
    else:
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers), normalizer

def add_lmc_paths(pt_files_per_run, num_points=10, lmc_dir="trainings/temp/"):
    os.makedirs(lmc_dir, exist_ok=True)
    lmc_runs = []

    # Get trajectory end:
    trajectory_ends = [torch.load(pt_files[-1], map_location='cpu', weights_only=True) for pt_files in pt_files_per_run]

    for i, _ in enumerate(pt_files_per_run):
        for j, _ in enumerate(pt_files_per_run):
            if i == j or i < j:
                continue
            path = linear_mode_connectivity_path(
                trajectory_ends[i],
                trajectory_ends[j],
                num_points)

            lmc_run = []

            for j, weights in enumerate(path):
                fname = os.path.join(lmc_dir, f"lmc_run{i}_{j}.pt")
                torch.save(weights, fname)
                lmc_run.append(fname)

            lmc_runs.append(lmc_run)

    return pt_files_per_run + lmc_runs


def linear_mode_connectivity_path(w1, w2, num_points=10):
    """
    Compute a straight‐line (LMC) path in weight‐space between two trained models.
    """
    # alphas from 0 to 1
    alphas = np.linspace(0.0, 1.0, num_points)

    # build the path: (1−α_i)*w1 + α_i*w2
    path = [(1 - a) * w1 + a * w2 for a in alphas]
    return path

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
    verbose=False,
    save_delta_pct: float = 0.01,       # <-- relative drop (e.g. 0.01 for 1%)
    avoid_overheat=False,
    step_lr_patience: int = 5,
    step_lr_factor: int = 0.5,
    last_saved_loss:int = None
):
    """
    Generic improved AE training loop with early stopping and scheduler.
    Saves best model to save_path.
    """
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import time

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=step_lr_factor, patience=step_lr_patience, threshold=1e-3, cooldown=0, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    epochs_no_improve = 0

    losses_log = []

    avg_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} ({avg_loss:.4f}, {scheduler.get_last_lr()[-1]:.3e})"):
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        losses_log.append({'epoch': epoch, 'loss': avg_loss})

        if verbose:
            print(f"Epoch {epoch:03d} - Avg Loss: {avg_loss:.6f}")

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0

            # first‐time save OR enough relative drop
            if last_saved_loss is None:
                do_save = True
            else:
                rel_drop = (last_saved_loss - avg_loss) / last_saved_loss
                do_save = (rel_drop >= save_delta_pct)

            if do_save:
                time.sleep(5)
                torch.save(model.state_dict(), save_path)
                last_saved_loss = avg_loss
                #if verbose:
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

        if avoid_overheat and epoch % 10 == 0:
            time.sleep(20)
            #if epoch % 20 == 0 and verbose:
            #    input(f"Epoch {epoch} complete. Press Enter to continue…")

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
        with torch.no_grad():
            model_repopulated = repopulate_model_fixed(model_flattened.clone(), model)
        #model_repopulated.eval()
        model_repopulated = model_repopulated.to(device)
        loss = loss_obj.get_loss(model_repopulated, loss_name, whichloss).detach()
        grid_losses.append(loss)

    grid_losses = torch.stack(grid_losses)
    return grid_losses

def compute_grid_losses_batched(
    grid_coords,
    transform,
    ae_model,
    model,
    loss_obj,
    loss_name,
    whichloss,
    device,
    batch_size=128,
    recalibrate_bn: bool = True,
    bn_recal_batches: int = 100
):
    """
    Compute losses over a latent grid in batches.

    Args:
      grid_coords: (N×2) tensor of z-coordinates.
      transform: NormalizeModelParameters instance.
      ae_model: your trained autoencoder.
      model: the “skeleton” model to repopulate.
      loss_obj, loss_name, whichloss: passed to get_loss().
      device: cpu / cuda.
      batch_size: how many grid points to decode at once.

    Returns:
      Tensor of shape (N,) with the loss for each grid point.
    """
    ae_model.eval()
    losses = []

    # We’ll iterate over the grid in chunks:
    for start in tqdm(range(0, grid_coords.size(0), batch_size), desc="Computing grid losses"):
        end = start + batch_size
        coords_batch = grid_coords[start:end].to(device)  # [B,2]

        # 1) decode entire batch at once
        with torch.no_grad():
            rec_batch = ae_model.decoder(coords_batch)     # [B, D]
        # 2) un-normalize
        rec_batch = rec_batch * transform.std.to(device) + transform.mean.to(device)

        # 3) repopulate & score one by one
        for flat_weights in rec_batch:
            # repopulate_model_fixed returns a new model instance
            model_i = repopulate_model_fixed(flat_weights.clone().cpu(), model)

            # 2) optional BN recalibration
            if recalibrate_bn:
                model.train()
                with torch.no_grad():
                    for batch_idx, (x, _) in enumerate(loss_obj.train_loader):
                        if batch_idx >= bn_recal_batches:
                            break
                        model(x.to(device))
                model.to(device).eval()

            model_i = model_i.to(device).eval()
            with torch.no_grad():
                loss_i = loss_obj.get_loss(model_i, loss_name, whichloss).detach()
            losses.append(loss_i)

    return torch.stack(losses)  # [N]


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
            loader = self.test_loader if loss_name == "test_loss" else self.train_loader

            total_loss = 0.0
            total_samples = 0
            criterion = torch.nn.CrossEntropyLoss(reduction='sum')

            with torch.no_grad():
                for data, target in loader:
                    output = dnn(data.to(self.device))
                    #loss += F.nll_loss(output, target.to(self.device), reduction='sum').item()
                    loss = criterion(output, target.to(self.device)).item()
                    total_loss += loss
                    total_samples += target.size(0)

            return torch.tensor(total_loss / total_samples)
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
    recalibrate_bn: bool = True,
    bn_recal_batches: int = 100
):
    """
    Computes:
    - trajectory_coordinates (2D latent space)
    - trajectory_models (decoded weights)
    - trajectory_losses (loss values)
    """
    if not recalibrate_bn:
        print(f"With recalibrate_bn=True, the normalization layers could be recalibrated for better loss values")
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
        # sanity-check size
        model_flattened = trajectory_models[i, :]
        total_params = sum(p.numel() for p in model.parameters())
        assert model_flattened.numel() == total_params, "Mismatch in parameter size."

        # 1) repopulate model weights
        with torch.no_grad():
            model = repopulate_model_fixed(model_flattened.clone(), model)

        # 2) optional BN recalibration
        if recalibrate_bn:
            model.train()
            with torch.no_grad():
                for batch_idx, (x, _) in enumerate(loss_obj.train_loader):
                    if batch_idx >= bn_recal_batches:
                        break
                    model(x.to(device))
            model.to(device).eval()

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

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import numpy as np

def plot_loss_landscape(
    xx, yy,
    grid_losses, trajectory_losses_list, trajectory_coords_list,
    rec_grid_models=None,
    draw_density=True,
    filled_contours=True,
    cmap='viridis',
    loss_label='Cross Entropy Loss',
    trajectory_labels=None,
    label_positions=None,
):
    # === PREPARE LOSSES ===
    grid_losses_pos = grid_losses.detach().cpu().numpy()

    # === SHARED COLOR SCALE ===
    traj_losses_all = np.concatenate([t for t in trajectory_losses_list])
    all_losses = np.concatenate([grid_losses_pos.flatten(), traj_losses_all])
    vmin = np.clip(all_losses.min() / 1.2, 1e-5, None)
    vmax = all_losses.max() * 1.2

    if vmin >= vmax or np.isclose(vmin, vmax):
        vmax = vmin * 10
        print(f"Adjusted nearly-constant losses: vmin={vmin}, vmax={vmax}")

    levels = np.logspace(np.log10(vmin), np.log10(vmax), 30)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # === BEGIN PLOTTING ===
    fig, ax = plt.subplots(figsize=(8, 6))

    # -- 1 Loss Landscape --
    X = xx.cpu().numpy()
    Y = yy.cpu().numpy()

    if filled_contours:
        contour = ax.contourf(X, Y, grid_losses_pos, levels=levels, norm=norm, cmap=cmap)
    else:
        contour = ax.contour(X, Y, grid_losses_pos, levels=levels, norm=norm, cmap=cmap)
        ax.clabel(contour, fmt="%.2e", fontsize=8)

    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    ticks = np.logspace(np.log10(vmin), np.log10(vmax), 5)  # customize number here
    cbar.set_ticks(ticks)
    cbar.ax.set_ylabel(loss_label, fontsize=12)

    # -- 2 & 3: Plot Multiple Trajectories --
    for z_tensor, losses_tensor in zip(trajectory_coords_list, trajectory_losses_list):
        z = z_tensor
        losses = losses_tensor
        # Lines
        for i in range(len(z) - 1):
            ax.plot([z[i, 0], z[i + 1, 0]], [z[i, 1], z[i + 1, 1]], color='k', linewidth=1)
        # Points
        ax.scatter(
            z[:, 0], z[:, 1],
            c=losses,
            cmap=cmap,
            norm=norm,
            s=40,
            edgecolors='k'
        )

    # ===== 3b: Annotate each trajectory at its last point =====
    offset_pts = 12  # how far, in points, to shift the label

    # defaults
    n_traj = len(trajectory_coords_list)
    if trajectory_labels is None:
        trajectory_labels = [f"traj {i}" for i in range(n_traj)]
    if label_positions is None:
        label_positions = ['auto'] * n_traj

    for idx, (z, losses, lab) in enumerate(zip(
            trajectory_coords_list,
            trajectory_losses_list,
            trajectory_labels)):
        x_end, y_end = float(z[-1, 0]), float(z[-1, 1])

        # decide alignment
        pos = label_positions[idx]
        if pos != 'auto':
            ha, va = pos
        else:
            dx = z[-1, 0] - z[-2, 0]
            dy = z[-1, 1] - z[-2, 1]
            ha = 'left'   if dx >= 0 else 'right'
            va = 'bottom' if dy >= 0 else 'top'

        # convert alignment into point‐offset direction
        ox =  offset_pts if ha == 'left'   else (-offset_pts if ha == 'right' else 0)
        oy =  offset_pts if va == 'bottom' else (-offset_pts if va == 'top'   else 0)

        # annotate with offset
        ax.annotate(
            lab,
            xy=(x_end, y_end),
            xytext=(ox, oy),
            textcoords='offset points',
            ha=ha, va=va,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6),
            arrowprops=dict(arrowstyle='-', lw=0)
        )

    # -- 4 OPTIONAL: Density Contours --
    if draw_density and rec_grid_models is not None:
        try:
            from NeuroVisualizer.neuro_aux.utils import get_density
            density = get_density(rec_grid_models.detach().cpu().numpy(), type='inverse', p=2)
            density = density.reshape(xx.shape)
            density_levels = np.logspace(
                np.log10(max(density.min(), 1e-3)),
                np.log10(density.max()),
                15
            )
            CS_density = ax.contour(
                X, Y, density,
                levels=density_levels,
                colors='white',
                linewidths=0.8
            )
            ax.clabel(CS_density, fmt=ticker.FormatStrFormatter('%.1f'), fontsize=7)
        except Exception as e:
            print("Density contour skipped:", e)

    # -- 5 Labels, Grid, Style --
    ax.set_title('Loss Landscape with Training Trajectory', fontsize=14)
    ax.set_xlabel('Latent Dimension 1', fontsize=12)
    ax.set_ylabel('Latent Dimension 2', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    # -- 6 Show --
    #plt.show()

    return fig


def compute_lmc_loss_path(
    weight_path: list[np.ndarray],
    base_model: torch.nn.Module,
    dataset_name: str,
    device: str = 'cpu',
    loss_name: str = 'test_loss',
    whichloss: str = 'crossentropy',
):
    """
    Given a straight‐line (LMC) path of flattened weight vectors, repopulate
    the model at each step and compute its loss.

    Args:
        weight_path: list of 1D numpy arrays (or torch tensors) of length = total_params
        base_model: uninitialized model architecture to clone for each step
        dataset_name: e.g. 'mnist', 'cifar10', 'cifar100'
        device: 'cpu' or 'cuda'
        loss_name: 'train_loss' or 'test_loss'
        whichloss: as expected by Loss.get_loss (e.g. 'crossentropy')

    Returns:
        losses: list of floats, same length as weight_path
    """
    # set up Loss evaluator
    loss_obj = Loss(dataset_name, device)

    losses = []
    for flat in tqdm(weight_path, desc="Compute Losses"):
        # ensure a torch tensor on CPU
        w = torch.as_tensor(flat, dtype=torch.float32, device=device)
        # make a fresh model copy and load weights
        model_i = deepcopy(base_model).to(device)
        repopulate_model_fixed(w, model_i)
        model_i.eval()

        # compute loss on the chosen split
        with torch.no_grad():
            loss_t = loss_obj.get_loss(model_i, loss_name, whichloss)
        losses.append(loss_t.item())

        

    return losses