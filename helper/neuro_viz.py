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
    step_lr_patience: int = 4,
    step_lr_factor: int = 0.5,
    last_saved_loss:int = None,
    max_reload_attempts: int = 2
):
    """
    Generic improved AE training loop with early stopping and scheduler.
    Saves best model to save_path.
    """
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import time

    reload_attempts = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=step_lr_factor, patience=step_lr_patience, threshold=1e-3, cooldown=0, min_lr=1e-7)
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
                if avoid_overheat:
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
            if os.path.exists(save_path) and reload_attempts < max_reload_attempts:
                print(f"⤴️  Reloading best checkpoint (attempt {reload_attempts + 1}/{max_reload_attempts})")
                state = torch.load(save_path, map_location=device)
                model.load_state_dict(state)
                model.to(device)
                epochs_no_improve = 0
                reload_attempts += 1
                continue

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

def _flatten_model_vec(m: torch.nn.Module) -> torch.Tensor:
    # If you have a project-specific flattener, call it here instead.
    from torch.nn.utils import parameters_to_vector
    return parameters_to_vector(m.parameters()).detach()

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
    bn_recal_batches: int = 100,
    bn_loader=None,
):
    """
    Returns (unchanged signature):
      trajectory_coordinates: [N, 2]
      trajectory_models:     [N, D] (decoded, de-normalized) == (2)_denorm
      trajectory_losses:     [N]    (task loss per repopulated model)
      ae_losses_decode:      [N]    (1)_norm vs (2)_norm  (AE recon on normalized inputs)
      ae_losses_finetuned:   [N]    (1)_denorm vs (3)_denorm  (**redefined** to your requested comparison)

    Internally (not returned unless you choose to):
      ae_losses_decode_denorm: [N]  (1)_denorm vs (2)_denorm
      ae_losses_finetuned_norm:[N]  (1)_norm   vs (3)_norm
    """
    if not recalibrate_bn:
        print("Tip: set recalibrate_bn=True to refresh BN stats for more accurate losses.")

    ae_model.eval()
    model = model.to(device)
    model_device = next(model.parameters()).device
    total_params = sum(p.numel() for p in model.parameters())

    mean = transform.mean.to(device)
    std  = transform.std.to(device)

    # ---- Decode trajectory + AE loss (no fine-tuning) ----
    trajectory_models, trajectory_coordinates = [], []
    ae_losses_decode = []              # (1)_norm vs (2)_norm
    orig_denorm_list = []              # store (1)_denorm aligned with samples

    with torch.no_grad():
        for batch in tqdm(trajectory_loader, desc="Decoding trajectory"):
            # batch: normalized flattened weights, shape [B, D]  == (1)_norm
            batch = batch.to(device)

            x_recon_norm, z = ae_model(batch)  # (2)_norm and latent coords
            # per-sample MSE over feature dim
            ae_mse = F.mse_loss(x_recon_norm, batch, reduction='none').mean(dim=1)  # [B]
            ae_losses_decode.append(ae_mse.cpu())

            # store coords + de-normalized decoded weights for repopulation
            trajectory_coordinates.append(z.cpu())
            x_recon_den = x_recon_norm * std + mean         # (2)_denorm
            x_orig_den  = batch * std + mean                # (1)_denorm
            orig_denorm_list.append(x_orig_den.cpu())
            trajectory_models.append(x_recon_den.cpu())

    trajectory_coordinates = torch.cat(trajectory_coordinates, dim=0)     # [N, 2]
    trajectory_models      = torch.cat(trajectory_models, dim=0)          # [N, D] (2)_denorm
    orig_denorm_all        = torch.cat(orig_denorm_list, dim=0)           # [N, D] (1)_denorm
    ae_losses_decode       = torch.cat(ae_losses_decode, dim=0).float()   # [N]

    # (optional extra metric): (1)_denorm vs (2)_denorm
    ae_losses_decode_denorm = ((trajectory_models - orig_denorm_all)**2).mean(dim=1).cpu().float()  # [N]

    print(f"✅ Decoded trajectory shapes: coords {trajectory_coordinates.shape}, models {trajectory_models.shape}")

    # BN recal source
    bn_src = bn_loader or getattr(loss_obj, "train_loader", None) or trajectory_loader

    # ---- Compute task losses + AE loss after repopulation ----
    trajectory_losses = []
    # redefine to your desired comparison: (1)_denorm vs (3)_denorm
    ae_losses_finetuned = []
    # optional normalized variant
    ae_losses_finetuned_norm = []

    for i in tqdm(range(trajectory_models.shape[0]), desc="Computing trajectory & AE(finetuned) losses"):
        flat_cpu = trajectory_models[i, :]  # (2)_denorm
        assert flat_cpu.numel() == total_params, "Mismatch in parameter size."

        with torch.no_grad():
            # repopulate model from decoded weights
            flat = flat_cpu.to(model_device, non_blocking=True)
            model = repopulate_model_fixed(flat, model)  # in-place or returns model

        # (optional) BN recalibration
        if recalibrate_bn and bn_src is not None:
            model.train()
            with torch.no_grad():
                for b_idx, batch in enumerate(bn_src):
                    x = batch[0] if (isinstance(batch, (list, tuple)) and len(batch) >= 1) else batch
                    model(x.to(model_device, non_blocking=True))
                    if b_idx + 1 >= bn_recal_batches:
                        break
            model.eval()

        # Task loss (on current repopulated model == stage 3 after optional BN)
        with torch.no_grad():
            loss_val = loss_obj.get_loss(model, loss_name, whichloss).item()
        trajectory_losses.append(loss_val)

        # Compare (1) original vs (3) finetuned in *denorm* space  -> your target metric
        with torch.no_grad():
            flat_ft = _flatten_model_vec(model).to(device)  # (3)_denorm, [D]
            mse_1v3_denorm = ((flat_ft.cpu() - orig_denorm_all[i])**2).mean().item()
            ae_losses_finetuned.append(mse_1v3_denorm)

            # optional normalized comparison
            flat_ft_norm = ((flat_ft - mean) / std).view(1, -1)          # (3)_norm
            orig_norm_i  = ((orig_denorm_all[i].to(device) - mean) / std).view(1, -1)  # (1)_norm
            mse_1v3_norm = F.mse_loss(flat_ft_norm, orig_norm_i, reduction='mean').item()
            ae_losses_finetuned_norm.append(mse_1v3_norm)

    trajectory_losses        = torch.tensor(trajectory_losses, dtype=torch.float32)        # [N]
    ae_losses_finetuned      = torch.tensor(ae_losses_finetuned, dtype=torch.float32)      # [N] (1)_denorm vs (3)_denorm

    print(f"AE loss changed (denorm) from {ae_losses_decode_denorm.mean():.4g} → {ae_losses_finetuned.mean():.4g}")

    # Keep original 5-tuple API (with redefined ae_losses_finetuned as requested)
    return trajectory_coordinates, trajectory_models, trajectory_losses, ae_losses_decode, ae_losses_finetuned

def repopulate_model_fixed(flattened_params, model):
    start_idx = 0
    for param in model.parameters():
        size = param.numel()
        sub_flattened = flattened_params[start_idx : start_idx + size].view(param.size())
        param.data.copy_(sub_flattened)
        start_idx += size
    return model


def compute_lmc_lines(
    pt_files_per_run,
    ae_model,
    transform,
    loss_obj,
    model,
    loss_name,
    whichloss,
    device,
    num_points=10,
    lmc_dir="trainings/temp/",
    batch_size=64,
    num_workers=0,
    recalibrate_bn=True,
    bn_recal_batches=100,
    bn_loader=None,
):
    """
    Constructs all Linear Mode Connectivity (LMC) paths between the *final checkpoints*
    of the given runs (via your existing add_lmc_paths), evaluates each path with your
    existing compute_trajectory, and returns lists of (coords, losses) per LMC path.

    Returns:
      lmc_coords_list:   [M] list of tensors with shape [num_points, 2]
      lmc_losses_list:   [M] list of tensors with shape [num_points]
      lmc_meta:          [M] list of dicts with metadata (source_run_i, source_run_j, pt_files)
    """
    # --- 1) Create LMC path files (uses your existing function) ---
    #     NOTE: add_lmc_paths returns pt_files_per_run + lmc_runs
    from torch.utils.data import Dataset

    original_runs_count = len(pt_files_per_run)
    all_runs = add_lmc_paths(
        pt_files_per_run, num_points=num_points, lmc_dir=lmc_dir
    )
    lmc_runs = all_runs[original_runs_count:]  # only the new LMC runs (each is a list of .pt files)

    # Small helper: dataset that yields *normalized flattened weights* expected by compute_trajectory
    class _WeightsDatasetFromPtFiles(Dataset):
        def __init__(self, pt_list, mean, std, device="cpu"):
            self.pt_list = pt_list
            self.mean = mean
            self.std = std
            self.device = device

        def __len__(self):
            return len(self.pt_list)

        def __getitem__(self, idx):
            # Load flattened weights (assumes checkpoints are flat or can be flattened elsewhere in your pipeline)
            w = torch.load(self.pt_list[idx], map_location="cpu")
            if isinstance(w, dict) and "flat" in w:
                flat = torch.as_tensor(w["flat"], dtype=torch.float32)
            else:
                # If stored as a 1D tensor already:
                flat = torch.as_tensor(w, dtype=torch.float32).view(-1)

            # Normalize to AE space
            flat = (flat.to(self.device) - self.mean) / self.std
            return flat

    lmc_coords_list = []
    lmc_losses_list = []
    lmc_meta = []

    # --- 2) For each LMC path, build a loader and reuse your compute_trajectory ---
    for path_idx, pt_list in enumerate(lmc_runs):
        ds = _WeightsDatasetFromPtFiles(
            pt_list,
            mean=transform.mean.to(device),
            std=transform.std.to(device),
            device=device,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

        traj_coords, traj_models, traj_losses, ae_losses_decode, ae_losses_finetuned = compute_trajectory(
            trajectory_loader=loader,
            ae_model=ae_model,
            transform=transform,
            loss_obj=loss_obj,
            model=model,
            loss_name=loss_name,
            whichloss=whichloss,
            device=device,
            recalibrate_bn=recalibrate_bn,
            bn_recal_batches=bn_recal_batches,
            bn_loader=bn_loader,
        )

        lmc_coords_list.append(traj_coords)   # [num_points, 2]
        lmc_losses_list.append(traj_losses)   # [num_points]
        lmc_meta.append({
            "path_index": path_idx,
            "pt_files": pt_list,
            # If you care which original runs formed this path, encode it in filenames in add_lmc_paths
            # (e.g., lmc_run{i}_{j}.pt). We pass the raw list back so you can parse if needed.
        })

    return lmc_coords_list, lmc_losses_list, lmc_meta

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
    lmc_coords_list=None,        # NEW
    lmc_losses_list=None         # NEW
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
    ticks = np.logspace(np.log10(vmin), np.log10(vmax), 5)
    cbar.set_ticks(ticks)
    cbar.ax.set_ylabel(loss_label, fontsize=12)

    # -- 2 & 3: Plot Training Trajectories --
    for z_tensor, losses_tensor in zip(trajectory_coords_list, trajectory_losses_list):
        z = z_tensor
        losses = losses_tensor
        for i in range(len(z) - 1):
            ax.plot([z[i, 0], z[i + 1, 0]], [z[i, 1], z[i + 1, 1]], color='k', linewidth=1)
        ax.scatter(
            z[:, 0], z[:, 1],
            c=losses, cmap=cmap, norm=norm,
            s=40, edgecolors='k'
        )

    # -- 3b: Annotate each trajectory at its last point --
    offset_pts = 5
    n_traj = len(trajectory_coords_list)
    if trajectory_labels is None:
        trajectory_labels = [f"traj {i}" for i in range(n_traj)]
    if label_positions is None:
        label_positions = ['auto'] * n_traj

    for idx, (z, losses, lab) in enumerate(zip(
            trajectory_coords_list, trajectory_losses_list, trajectory_labels)):
        x_end, y_end = float(z[-1, 0]), float(z[-1, 1])
        pos = label_positions[idx]
        if pos != 'auto':
            ha, va = pos
        else:
            dx = z[-1, 0] - z[-2, 0]
            dy = z[-1, 1] - z[-2, 1]
            ha = 'left'   if dx >= 0 else 'right'
            va = 'bottom' if dy >= 0 else 'top'
        ox = offset_pts if ha == 'left' else (-offset_pts if ha == 'right' else 0)
        oy = offset_pts if va == 'bottom' else (-offset_pts if va == 'top' else 0)
        ax.annotate(
            lab, xy=(x_end, y_end), xytext=(ox, oy),
            textcoords='offset points', ha=ha, va=va,
            fontsize=7, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6),
            arrowprops=dict(arrowstyle='-', lw=0)
        )

    # -- 4: Plot LMC Lines (in red) --
    if lmc_coords_list is not None and lmc_losses_list is not None:
        for z_tensor, losses_tensor in zip(lmc_coords_list, lmc_losses_list):
            z = z_tensor
            losses = losses_tensor
            ax.plot(z[:, 0], z[:, 1], color='red', linewidth=1.5, linestyle='--')
            ax.scatter(
                z[:, 0], z[:, 1],
                c=losses, cmap=cmap, norm=norm,
                s=30, edgecolors='r'
            )

    # -- 5 OPTIONAL: Density Contours --
    if draw_density and rec_grid_models is not None:
        try:
            from NeuroVisualizer.neuro_aux.utils import get_density
            density = get_density(rec_grid_models.detach().cpu().numpy(), type='inverse', p=2)
            density = density.reshape(xx.shape)
            density_levels = np.logspace(
                np.log10(max(density.min(), 1e-3)), np.log10(density.max()), 15
            )
            CS_density = ax.contour(
                X, Y, density,
                levels=density_levels,
                colors='white', linewidths=0.8
            )
            ax.clabel(CS_density, fmt=ticker.FormatStrFormatter('%.1f'), fontsize=7)
        except Exception as e:
            print("Density contour skipped:", e)

    # -- 6 Labels, Grid, Style --
    ax.set_title('Loss Landscape with Training Trajectories and LMC Paths', fontsize=14)
    ax.set_xlabel('Latent Dimension 1', fontsize=12)
    ax.set_ylabel('Latent Dimension 2', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

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


def plot_density_only(
    xx, yy, rec_grid_models,
    trajectory_coords_list=None,
    trajectory_losses_list=None,
    cmap="plasma",
    point_cmap="plasma",
    point_size=40,
):
    """
    Plot density (filled contours) and optionally overlay training trajectories.
      - Lines: black
      - Points: colored by per-step losses if provided, else white

    Args:
        xx, yy: meshgrid tensors from Neuro-Visualizer (same shape)
        rec_grid_models: decoded models on the grid (for density)
        trajectory_coords_list: list of [T_i, 2] tensors (optional)
        trajectory_losses_list: list of [T_i] arrays/tensors (optional)
    """
    X = xx.cpu().numpy()
    Y = yy.cpu().numpy()

    try:
        from NeuroVisualizer.neuro_aux.utils import get_density

        # --- Density ---
        density = get_density(rec_grid_models.detach().cpu().numpy(), type='inverse', p=2)
        density = density.reshape(xx.shape)

        # Guard for LogNorm
        dmin = max(float(np.nanmin(density)), 1e-8)
        dmax = float(np.nanmax(density))
        if not np.isfinite(dmax) or dmax <= dmin:
            raise ValueError(f"Invalid density range: dmin={dmin}, dmax={dmax}")

        levels = np.logspace(np.log10(dmin), np.log10(dmax), 50)

        fig, ax = plt.subplots(figsize=(7, 6))
        CS = ax.contourf(
            X, Y, density,
            levels=levels,
            cmap=cmap,
            norm=LogNorm(vmin=dmin, vmax=dmax)
        )

        cbar = plt.colorbar(CS, ax=ax, shrink=0.8)
        cbar.set_label("Density", fontsize=12)
        cbar.set_ticks(np.logspace(np.log10(dmin), np.log10(dmax), 5))

        # --- Trajectories (optional) ---
        if trajectory_coords_list is not None and len(trajectory_coords_list) > 0:
            for idx, z_tensor in enumerate(trajectory_coords_list):
                z = np.asarray(z_tensor)
                # lines
                if len(z) >= 2:
                    ax.plot(z[:, 0], z[:, 1], color='k', linewidth=1.2, alpha=0.9, zorder=3)
                # points (loss-colored if provided)
                if trajectory_losses_list is not None and idx < len(trajectory_losses_list) and trajectory_losses_list[idx] is not None:
                    losses = np.asarray(trajectory_losses_list[idx])
                    # robust positive norm for coloring (LogNorm ok if losses>0; else fallback linear)
                    if np.all(np.isfinite(losses)) and np.nanmin(losses) > 0:
                        pn = LogNorm(vmin=max(np.nanmin(losses), 1e-8), vmax=np.nanmax(losses))
                        sc = ax.scatter(
                            z[:, 0], z[:, 1],
                            c=losses, cmap=point_cmap, norm=pn,
                            s=point_size, edgecolors='k', linewidths=0.5, zorder=4
                        )
                        # one shared colorbar for the last scatter added
                        cb2 = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
                        cb2.set_label("Trajectory loss", fontsize=10)
                    else:
                        ax.scatter(z[:, 0], z[:, 1], s=point_size, c='white', edgecolors='k', linewidths=0.5, zorder=4)
                else:
                    ax.scatter(z[:, 0], z[:, 1], s=point_size, c='white', edgecolors='k', linewidths=0.5, zorder=4)

        ax.set_title("Density Plot (+ Trajectories)", fontsize=14)
        ax.set_xlabel("Latent Dimension 1", fontsize=12)
        ax.set_ylabel("Latent Dimension 2", fontsize=12)

        return fig

    except Exception as e:
        print("Density plot skipped:", e)
        return None