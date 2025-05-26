import math
from collections import deque
from pprint import pprint

import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
from torch.nn import CrossEntropyLoss
from IPython.display import clear_output
import logging
import ipywidgets as widgets
from IPython.display import display
import torch.nn.functional as F

marker_styles = ['o', 'p', '^', 'X', 'D', 'P', 'v', '<', '>', '*', "s"]

def train_model_with_embedding_tracking(
    model, train_loader, test_loader, subset_loader, device, num_classes,
    epochs=10, learning_rate=0.001, embedding_records_per_epoch=10, average_window_size=30,
    track_gradients=True, track_embedding_drift=True, track_cosine_similarity=False, track_scheduled_lr=False,
    track_pca=False, early_stopping=True, patience=4, weight_decay=0.05, optimizer=None, scheduler=None,
):
    assert model.__class__.__name__ in ['ViT', 'CNN', 'MLP'], "Model must be ViT, CNN, or MLP"
    optimizer, scheduler, criterion = _setup_training(model, learning_rate, epochs, weight_decay, optimizer=optimizer, scheduler=scheduler)

    # Initialize lists for performance tracking
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    scheduler_history = []

    # Initialize lists for embedding snapshot
    num_batches = len(train_loader)
    embedding_batch_interval = math.ceil(num_batches / embedding_records_per_epoch)
    print(f"{num_batches} Batches, {embedding_records_per_epoch} Records per Epoch, Resulting Batch interval: {embedding_batch_interval}")
    embedding_snapshots, embedding_snapshot_labels, embedding_indices = [], [], []
    embedding_counter = 0

    # Initialize lists for gradient tracking
    gradient_norms, max_gradients, grad_param_ratios, gradient_indices = [], [], [], []
    gradient_counter = 0 # will track absolute batch index for x-axis

    # Logging setup
    logging.basicConfig(level=logging.INFO, force=True)
    log_history = []

    # Visualization setup
    num_figures = 1 + int(track_gradients) + int(track_embedding_drift) + int(track_cosine_similarity) + int(
        track_pca) + int(track_scheduled_lr)
    backend = matplotlib.get_backend().lower()
    if 'widget' in backend:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()
    else:
        fig = ax1 = ax2 = None  # placeholder

    # Early stopping setup
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss, correct_train, total_train = 0, 0, 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient Tracking
            if track_gradients and (batch_idx % int(embedding_batch_interval/2) == 0):
                grad_norm, max_grad, grad_ratio = _track_gradients(model)
                gradient_norms.append(grad_norm)
                max_gradients.append(max_grad)
                grad_param_ratios.append(grad_ratio)
                gradient_indices.append(gradient_counter)
                gradient_counter += 1

            optimizer.step()

            # Embedding Tracking
            if batch_idx % embedding_batch_interval == 0:
                model.eval()
                with torch.no_grad():
                    batch_embeddings, batch_labels = [], []
                    for data_sub, target_sub in subset_loader:
                        data_sub = data_sub.to(device)
                        _, emb = model(data_sub, return_embedding=True)
                        batch_embeddings.append(emb.cpu().numpy())
                        batch_labels.append(np.array(target_sub.flatten()))
                    embedding_snapshots.append(np.concatenate(batch_embeddings, axis=0))
                    embedding_snapshot_labels.append(np.concatenate(batch_labels, axis=0))

                if track_embedding_drift:
                    embedding_drifts = _calculate_embedding_drift(embedding_snapshots)
                model.train()
                embedding_indices.append(embedding_counter)
                embedding_counter += 1


            # Training metrics
            epoch_train_loss += loss.item()
            _, preds = torch.max(output, dim=1)
            correct_train += (preds == target).sum().item()
            total_train += target.size(0)

        # === Epoch-wise accuracy ===
        train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # === Validation phase ===
        model.eval()
        epoch_val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, embedding = model(data, return_embedding=True)
                loss = criterion(output, target)
                epoch_val_loss += loss.item()
                _, preds = torch.max(output, dim=1)
                correct_val += (preds == target).sum().item()
                total_val += target.size(0)

        val_loss = epoch_val_loss / len(test_loader)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if scheduler is not None:
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            scheduler_history.append(scheduler.get_last_lr()[-1])
        else:
            scheduler_history.append(learning_rate)

        # Live plot update
        fig, axs = _live_plot_update(num_figures=num_figures)
        _plot_loss_accuracy(axs[0], epoch, epochs, train_losses, val_losses, train_accuracies, val_accuracies)

        pos = 1
        if track_gradients:
            _plot_gradients(axs[pos], gradient_indices, gradient_norms, max_gradients, grad_param_ratios, average_window_size)
            pos += 1
        if track_scheduled_lr and scheduler is not None:
            _plot_scheduled_lr(axs[pos], scheduler_history)
            pos += 1
        if track_embedding_drift:
            _plot_embedding_drift(axs[pos], embedding_drifts)
            pos += 1
        if track_pca:
            _plot_pca(axs[pos], embedding_snapshots, embedding_snapshot_labels, embedding_records_per_epoch, num_classes=num_classes)
            pos += 1
        if track_cosine_similarity:
            # Todo: Implement cosine similarity tracking
            pos += 1

        plt.tight_layout()
        plt.show()

        # Print summary
        log_line = (
            f"Epoch [{epoch + 1}/{epochs}] "
            f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )
        log_history.append(log_line)
        print("\n".join(log_history))

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            print(f"New best validation loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")

        # Check if patience limit reached
        if early_stopping and patience_counter >= patience:
            log_line=f"Early stopping triggered at epoch {epoch + 1}"
            log_history.append(log_line)
            print(log_line)
            break

    print(
        f"Recorded {len(embedding_snapshots)} embeddings in {epochs} epochs "
        f"({len(embedding_snapshots) / epochs:.2f} per epoch)."
    )

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'subset_embeddings': embedding_snapshots,
        'subset_labels': embedding_snapshot_labels,
        'embedding_drifts': embedding_drifts,
        'gradient_norms': gradient_norms,
        'max_gradients': max_gradients,
        'grad_param_ratios': grad_param_ratios,
        'scheduler_history': scheduler_history,
    }

def _setup_training(model, learning_rate, epochs, weight_decay, optimizer=None, scheduler=None):

    assert model.__class__.__name__ in ['ViT', 'CNN', 'MLP'], "Model must be ViT, CNN, or MLP"

    if model.__class__.__name__ == 'MLP':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) if optimizer is None else optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) if scheduler is None else scheduler
    elif model.__class__.__name__ == 'CNN':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) if optimizer is None else optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if scheduler is None else scheduler
    elif model.__class__.__name__ == 'ViT':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) if optimizer is None else optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if scheduler is None else scheduler

    criterion = CrossEntropyLoss()
    return optimizer, scheduler, criterion

def _live_plot_update(num_figures=1, ncols=2):
    plt.close('all')
    clear_output(wait=True)
    nrows = math.ceil(num_figures / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 4*nrows))
    return fig, axs.flatten()

def _track_gradients(model):
    """Tracks gradient norms and parameter ratios."""
    total_norm = 0.0
    max_grad = 0.0
    ratios = []

    for p in model.parameters():
        if p.grad is not None:
            grad = p.grad.detach()
            total_norm += grad.norm(2).item() ** 2
            max_grad = max(max_grad, grad.abs().max().item())
            if p.data.norm() > 0:
                ratios.append((grad.norm() / p.data.norm()).item())

    return total_norm ** 0.5, max_grad, np.mean(ratios) if ratios else np.nan

def _calculate_embedding_drift(embedding_snapshots, max_power=5):
    """
    Calculate embedding drift based on the snapshots.
    Drift is calculated as the mean Euclidean distance between snapshots.
    Uses skip steps as powers of 2 (i.e., 1, 2, 4, 8, ...).
    """
    drifts = {2**n: [] for n in range(max_power)}

    # Iterate over all snapshots
    for i in range(1, len(embedding_snapshots)):
        current_snapshot = embedding_snapshots[i]

        # Compare with previous snapshots using 2^n steps
        for n in range(max_power):
            skip = 2**n
            if i - skip >= 0:
                previous_snapshot = embedding_snapshots[i - skip]
                drift = np.linalg.norm(current_snapshot - previous_snapshot, axis=1).mean()
                drifts[skip].append(drift)
            else:
                drifts[skip].append(np.nan)

    return drifts

def _plot_loss_accuracy(ax, epoch, epochs, train_losses, val_losses, train_accuracies, val_accuracies):
    """Plots loss and accuracy over epochs."""
    ax.set_title("Loss & Accuracy per Epoch")
    epochs_range = range(1, epoch + 2)
    ax.plot(epochs_range, train_losses, 'b-', label='Train Loss', alpha=0.7)
    ax.plot(epochs_range, val_losses, 'r-', label='Val Loss', alpha=0.7)
    ax.set_ylabel('Loss')
    ax.legend(loc='upper left')
    #ax.set_xlim(1, epochs)  # Fixed x-axis
    ax.set_ylim(0, max(val_losses + train_losses))  # Fixed range for loss
    #ax.set_xticks(list(range(1, epochs + 1)))  # Integer ticks only

    ax2 = ax.twinx()
    ax2.plot(epochs_range, train_accuracies, 'g--', label='Train Acc', alpha=0.7)
    ax2.plot(epochs_range, val_accuracies, 'orange', linestyle='--', label='Val Acc', alpha=0.7)
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(min(min(train_accuracies), min(val_accuracies)) * 0.9, 1.0)
    ax2.legend(loc='upper right')

def _plot_gradients(ax, batch_indices, gradient_norms, max_gradients, grad_param_ratios, window_size):
    """Plots gradient norms and ratios."""
    ax.set_title("Gradient Metrics")
    ax.plot(batch_indices, max_gradients, color='orange', label='Max Gradient', alpha=0.3)
    ax.plot(batch_indices, grad_param_ratios, color='red', label='Grad/Param Ratio', alpha=0.3)

    avg_indices = batch_indices[window_size - 1:]
    if len(batch_indices) >= window_size:
        ax.plot(avg_indices, _moving_average(max_gradients, window_size), color='orange',
                label='Max Gradient (Avg)', alpha=0.8, linewidth=1.5)
        ax.plot(avg_indices, _moving_average(grad_param_ratios, window_size), color='red',
                label='Grad/Param Ratio (Avg)', alpha=0.8, linewidth=1.5)
    ax.set_ylim(0, max(max(max_gradients), max(grad_param_ratios)))
    ax.legend(loc='upper left')

    ax2 = ax.twinx()
    ax2.plot(batch_indices, gradient_norms, 'blue', linestyle='--', label='Gradient Norm', alpha=0.3)
    if len(batch_indices) >= window_size:
        ax2.plot(avg_indices, _moving_average(gradient_norms, window_size), color='blue', linestyle='-',
                 label='Gradient Norm (Avg)', alpha=0.8, linewidth=1.5)
    ax2.set_ylim(0, max(gradient_norms))
    ax2.set_ylabel('Gradient Norm')
    ax2.legend(loc='upper right')


from scipy.stats import pearsonr

def _plot_scheduled_lr(ax, scheduler_history):
    ax.plot(scheduler_history, label='Learning Rate')
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")

def _plot_pca(ax, embedding_snapshots, embedding_snapshot_labels, embedding_records_per_epoch, out_dim=2, num_classes=10, size=10):
    """Plots the PCA of the embeddings in the last epoch."""
    from sklearn.decomposition import PCA

    window_data = np.concatenate(embedding_snapshots[-embedding_records_per_epoch:], axis=0)
    reducer = PCA(n_components=out_dim)
    reducer.fit(window_data)

    projection = reducer.transform(embedding_snapshots[-1])

    # Scatter Plot of projection
    labels = embedding_snapshot_labels[-1]
    if num_classes <= 10:
        cmap = 'tab10'
        ax.scatter(projection[:, 0], projection[:, 1],
                   c=labels, cmap=cmap, alpha=0.7, s=size)
    else:
        cmap = plt.get_cmap('tab20', num_classes)
        for i in range(num_classes):
            idx = labels == i
            marker = marker_styles[i % len(marker_styles)]
            color = cmap(i / num_classes)

            ax.scatter(projection[idx, 0], projection[idx, 1],
                       c=[color], marker=marker, label=str(i),
                       alpha=0.7, edgecolors='none', s=size)

def visualization_drift_vs_embedding_drift(projections, embedding_drifts, verbose=True, embeddings=False):
    """
    Visualizes the drift of embeddings and computes the Pearson correlation between
    visualization drift and high-dimensional embedding drift for each data series.

    Args:
        projections (list of np.ndarray): Low-dimensional projections generated using
            dimensionality reduction techniques (e.g., t-SNE or UMAP).
        embedding_drifts (dict): Dictionary of high-dimensional embedding drift values,
            where each key represents a skip step (e.g., 1, 2, 4, ...) and each value is an
            array of drift measurements.
        verbose (bool): If True, prints the correlation values for each series and the mean correlation.
        embeddings (bool): If True, compute the embedding drift per each series.

    Returns:
        float: Mean Pearson correlation between visualization drift and embedding drift
            across all data series.
    """
    if embeddings:
        embedding_drifts = _calculate_embedding_drift(embedding_drifts)
    else:
        embedding_drifts = embedding_drifts.copy()

    # Use the existing function to calculate visualization drift
    visualization_drifts = _calculate_embedding_drift(projections)

    # Ensure embedding_drifts and visualization_drifts have the same length
    assert len(embedding_drifts) == len(visualization_drifts), "Mismatch in drift lengths."

    # Calculate correlation per data series and store in a list
    correlations = []
    for i in embedding_drifts.keys():
        emb_drift = np.asarray(embedding_drifts[i]).flatten()
        vis_drift = np.asarray(visualization_drifts[i]).flatten()

        # Check if lengths match
        assert len(emb_drift) == len(vis_drift), f"Mismatch in series {i}: {len(emb_drift)} vs {len(vis_drift)}"

        valid_mask = ~np.isnan(emb_drift) & ~np.isnan(vis_drift)
        emb_drift = emb_drift[valid_mask]
        vis_drift = vis_drift[valid_mask]

        # Calculate Pearson correlation
        correlation, _ = pearsonr(emb_drift, vis_drift)
        correlations.append(correlation)
        if verbose:
            print(f"Series {i} - Correlation: {correlation:.4f}")

    # Calculate and print mean correlation
    mean_correlation = np.mean(correlations)
    if verbose:
        print(f"Mean Correlation: {mean_correlation:.4f}")

        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        _plot_embedding_drift(axs[0], visualization_drifts, title="Visualization Drift")
        _plot_embedding_drift(axs[1], embedding_drifts)

        plt.legend()
        plt.show()

    return mean_correlation

def _plot_embedding_drift(ax, embedding_drifts, title="Embedding Drift", max_multiply=1.1):
    """Plots embedding drift."""
    colors = ['green', 'blue', 'orange', 'red', 'purple']
    labels = ['Drift 1', 'Drift 2', 'Drift 4', 'Drift 8', 'Drift 16']

    y_max = 0
    for skip, color, label in zip(embedding_drifts.keys(), colors, labels):
        drift_data = embedding_drifts[skip]
        indices = range(1, len(drift_data) + 1)
        ax.plot(indices, drift_data, color=color, label=label, alpha=0.7)
        if len(drift_data[skip:]) > 0:
            y_max = max(y_max, max(drift_data[skip:]))
    ax.set_ylim(0, y_max * max_multiply)
    ax.set_title(title)
    ax.set_ylabel("Drift Distance")
    ax.set_xlabel("Snapshot Index")
    ax.legend(loc='upper right')

def _moving_average(data, window_size):
    """Calculate the moving average with a specified window size."""
    if len(data) < window_size:
        return data  # Not enough data points to calculate the average
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def generate_projections(
    embeddings_list,
    method='tsne',
    pca_fit_basis='first',
    max_frames=None,
    reverse_computation=False,
    random_state=42,
    tsne_init='pca', #'pca' or 'random'
    tsne_perplexity=30.0, # often between 5–50
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    metric='euclidean', # for umap and tsne
    window_size=10,
    out_dim=2 #2D vs 3D
):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    import numpy as np

    assert method in ['tsne', 'pca', 'umap']
    assert pca_fit_basis in ['first', 'last', 'all', 'last_visualized', 'window', int]

    projections = []
    max_frames = max_frames or len(embeddings_list)

    # Determine basis data
    if isinstance(pca_fit_basis, int):
        basis_data = embeddings_list[pca_fit_basis]
    elif pca_fit_basis == 'first':
        basis_data = embeddings_list[0]
    elif pca_fit_basis == 'last':
        basis_data = embeddings_list[-1]
    elif pca_fit_basis == 'last_visualized':
        basis_data = embeddings_list[max_frames - 1]
    elif pca_fit_basis == 'all':
        basis_data = np.concatenate(embeddings_list, axis=0)
    elif pca_fit_basis == 'window':
        basis_data = embeddings_list[0]
    else:
        raise ValueError(f"Invalid pca_fit_basis: {pca_fit_basis}")

    if method == 'pca' and pca_fit_basis == 'window':
        from scipy.linalg import orthogonal_procrustes

        prev_components = None
        prev_projection = None

        for i in range(max_frames):
            window_start = max(0, i - window_size + 1)
            window_data = np.concatenate(embeddings_list[window_start:i+1], axis=0)
            reducer = PCA(n_components=out_dim)
            reducer.fit(window_data)

            # Flip correction
            if prev_components is not None:
                for j in range(out_dim):
                    if np.dot(reducer.components_[j], prev_components[j]) < 0:
                        reducer.components_[j] *= -1

            # Transform and align the projection
            projection = reducer.transform(embeddings_list[i])

            if prev_projection is not None:
                # Align projection using Procrustes (orthogonal) alignment
                R, _ = orthogonal_procrustes(projection, prev_projection)
                projection = projection @ R

            projections.append(projection)
            prev_components = reducer.components_.copy()
            prev_projection = projection.copy()

    elif method == 'pca':
        reducer = PCA(n_components=out_dim)
        reducer.fit(basis_data)
        for i in range(max_frames):
            projections.append(reducer.transform(embeddings_list[i]))

    elif method == 'tsne':
        tsne = TSNE(n_components=out_dim, init=tsne_init, perplexity=tsne_perplexity, random_state=random_state, metric=metric)
        tsne.fit(basis_data)
        if reverse_computation:
            projections = [None] * max_frames
            projections[-1] = tsne.fit_transform(embeddings_list[max_frames - 1])
            for i in range(max_frames - 2, -1, -1):
                tsne = TSNE(n_components=out_dim, init=projections[i + 1], perplexity=tsne_perplexity, random_state=random_state, metric=metric)
                projections[i] = tsne.fit_transform(embeddings_list[i])
        else:
            projections.append(tsne.fit_transform(embeddings_list[0]))
            for i in range(1, max_frames):
                tsne = TSNE(n_components=out_dim, init=projections[-1], perplexity=tsne_perplexity, random_state=random_state, metric=metric)
                projections.append(tsne.fit_transform(embeddings_list[i]))

    elif method == 'umap':
        reducer = umap.UMAP(n_components=out_dim, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, metric=metric)
        if reverse_computation:
            for i in range(max_frames - 1, -1, -1):
                reducer.fit(embeddings_list[i])
                projection = reducer.transform(embeddings_list[i])
                projections.insert(0, projection)
        else:
            reducer.fit(basis_data)
            for i in range(max_frames):
                projection = reducer.transform(embeddings_list[i])
                projections.append(projection)

    return projections


def denoise_projections(projections, blend=0.5, window_size=5, mode='window'):
    """
    Smooths projections using either causal or window-based blending.

    Args:
        projections (list of np.ndarray): List of (n_samples, n_dims) arrays over time.
        blend (float): Blend factor (0 = original, 1 = fully smoothed).
        window_size (int): Number of steps for averaging (only used in 'window' mode).
        mode (str): 'exponential' (blend with previous) or 'window' (blend with surrounding).

    Returns:
        list of np.ndarray: Smoothed projections.
    """
    assert 0 <= blend <= 1, "Blend must be in [0, 1]"
    assert mode in ['exponential', 'window'], "Mode must be 'causal' or 'window'"

    num_frames = len(projections)
    denoised_projections = []

    for i in range(num_frames):
        if mode == 'exponential' and i > 0:
            ref = denoised_projections[-1]
        elif mode == 'window':
            start = max(0, i - window_size)
            end = min(num_frames, i + 1)
            ref = np.mean(projections[start:end], axis=0)
        else:
            ref = projections[i]  # fallback for first frame in 'exponential' mode

        blended = (1 - blend) * projections[i] + blend * ref
        denoised_projections.append(blended)

    return denoised_projections

def animate_projections(
    projections,
    labels,
    interpolate=False,
    steps_per_transition=10,
    frame_interval=50,
    figsize=(4, 4),
    dot_size=5,
    alpha=0.7,
    cmap='tab10',
    title_base='Embedding Evolution',
    axis_lim=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Interpolate if requested
    if interpolate:
        projections_interp = []
        for a, b in zip(projections[:-1], projections[1:]):
            for alpha_step in np.linspace(0, 1, steps_per_transition, endpoint=False):
                interp = (1 - alpha_step) * a + alpha_step * b
                projections_interp.append(interp)
        projections_interp.append(projections[-1])
    else:
        projections_interp = projections

    # Auto axis limit
    if axis_lim is None:
        all_proj = np.concatenate(projections_interp, axis=0)
        max_abs = np.max(np.abs(all_proj))
        axis_lim = max_abs * 1.0

    # Plot setup
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter([], [], s=dot_size, c=[], cmap=cmap, alpha=alpha)
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        scatter.set_offsets(projections_interp[frame])
        scatter.set_array(np.array(labels).flatten())
        return scatter,

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(projections_interp),
        interval=frame_interval,
        blit=True
    )

    return ani

def show_with_slider(
        projections,
        labels,
        figsize=(5, 5),
        dot_size=5,
        alpha=0.5,
        interpolate=False,
        steps_per_transition=10,
        dataset=None,
        show_legend=False,
        symmetric=False
):
    from vision_classification import get_text_labels
    class_names = range(0, 100) if dataset is None else get_text_labels(dataset)

    projections = np.array(projections)
    projections = _interpolate_projections(projections, steps_per_transition) if interpolate else projections

    if dataset == "cifar100":
        fine_index_to_plot_config, fine_to_coarse = _prepare_cifar100_plot_config(class_names)

    # Handle filtered labels robustly
    unique_labels = np.unique(np.concatenate(labels))
    num_classes = len(unique_labels)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=figsize)

    # Axis limits
    x_min, x_max, y_min, y_max = _set_equal_aspect_limits(projections, ax, symmetric)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)

    # Plot initial frame (0)
    def draw_frame(idx):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)

        projection = projections[idx]
        label_frame = labels[0]

        if dataset == "cifar100":
            for fine_idx in unique_labels:
                idxs = label_frame == fine_idx
                config = fine_index_to_plot_config.get(fine_idx, {})
                color = config.get('color', 'gray')
                marker = config.get('marker', 'o')
                ax.scatter(projection[idxs, 0], projection[idxs, 1],
                           c=[color], marker=marker, label=class_names[fine_idx],
                           alpha=alpha, edgecolors='none', s=dot_size * 3)

        else:
            cmap_used = 'tab10' if num_classes <= 10 else 'tab20'
            # Map filtered labels to 0-based indices for colormap
            mapped_labels = np.array([label_to_index[lbl] for lbl in label_frame])
            sc = ax.scatter(projection[:, 0], projection[:, 1],
                            c=mapped_labels, cmap=cmap_used, alpha=alpha, s=dot_size)
            handles = [mlines.Line2D([], [], color=sc.cmap(sc.norm(i)), marker='o', linestyle='None',
                                     markersize=6, label=f"{lbl}: {class_names[lbl]}") for i, lbl in
                       enumerate(unique_labels)]
            if show_legend:
                ax.legend(handles=handles, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

        if show_legend and dataset == "cifar100":
            _create_cifar100_legend(ax, fine_index_to_plot_config, unique_labels, class_names, fine_to_coarse)

    # Initial draw
    draw_frame(0)

    # Slider and update
    def update(frame_idx):
        draw_frame(frame_idx)

    slider = widgets.Play(min=0, max=len(projections) - 1, step=1)
    slider_control = widgets.IntSlider(min=0, max=len(projections) - 1, step=1)
    widgets.jslink((slider, 'value'), (slider_control, 'value'))

    out = widgets.interactive_output(update, {'frame_idx': slider_control})
    display(widgets.VBox([widgets.HBox([slider, slider_control]), out]))


def show_multiple_projections_with_slider(
    projections_list,
    labels,
    titles=None,
    figsize_per_plot=(5, 5),
    dot_size=5,
    alpha=0.6,
    interpolate=False,
    steps_per_transition=10,
    shared_axes=True,
    dataset=None
):
    from vision_classification import get_text_labels, get_cifar100_fine_to_coarse_labels, get_cifar100_coarse_to_fine_labels
    class_names = range(0, 100) if dataset is None else get_text_labels(dataset)

    if dataset == "cifar100":
        fine_index_to_plot_config, fine_to_coarse = _prepare_cifar100_plot_config(class_names)

    projections_list = [np.array(p) for p in projections_list]

    # Interpolation

    if interpolate:
        projections_list = [_interpolate_projections(p, steps_per_transition) for p in projections_list]

    # Check length
    n_frames = len(projections_list[0])
    for p in projections_list:
        assert len(p) == n_frames, "All projection sets must have the same number of frames"

    num_views = len(projections_list)
    unique_labels = np.unique(np.concatenate(labels))
    label_frame = labels[0]

    # Layout
    if num_views <= 4:
        nrows, ncols = 1, num_views
    else:
        ncols = math.ceil(math.sqrt(num_views))
        nrows = math.ceil(num_views / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes[num_views:]:
        fig.delaxes(ax)
    axes = axes[:num_views]

    # Axis limits
    axis_limits = _compute_axis_limits(projections_list, shared=shared_axes)

    # Create scatter handles
    scatters = []
    for i, ax in enumerate(axes):
        ax.set_xlim(axis_limits[i])
        ax.set_ylim(axis_limits[i])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        title = titles[i] if titles and i < len(titles) else f"View {i+1}"
        ax.set_title(title)

        if dataset == "cifar100":
            scatter_dict = {}
            for fine_idx in unique_labels:
                idxs = label_frame == fine_idx
                color = fine_index_to_plot_config[fine_idx]['color']
                marker = fine_index_to_plot_config[fine_idx]['marker']
                scatter = ax.scatter(projections_list[i][0][idxs, 0],
                                     projections_list[i][0][idxs, 1],
                                     c=[color], marker=marker,
                                     alpha=alpha, edgecolors='none', s=dot_size * 2)
                scatter_dict[fine_idx] = scatter
            scatters.append(scatter_dict)
        else:
            sc = ax.scatter(projections_list[i][0][:, 0], projections_list[i][0][:, 1],
                            c=label_frame, cmap='tab10', s=dot_size, alpha=alpha)
            scatters.append(sc)

    # Update logic
    def update(frame_idx):
        for i in range(num_views):
            if dataset == "cifar100":
                for fine_idx in unique_labels:
                    idxs = label_frame == fine_idx
                    scatters[i][fine_idx].set_offsets(projections_list[i][frame_idx][idxs])
            else:
                scatters[i].set_offsets(projections_list[i][frame_idx])
                scatters[i].set_array(np.array(label_frame))
        fig.canvas.draw_idle()

    # Slider
    slider = widgets.Play(min=0, max=n_frames - 1, step=1)
    slider_control = widgets.IntSlider(min=0, max=n_frames - 1, step=1)
    widgets.jslink((slider, 'value'), (slider_control, 'value'))

    out = widgets.interactive_output(update, {'frame_idx': slider_control})
    display(widgets.VBox([widgets.HBox([slider, slider_control]), out]))

def _set_equal_aspect_limits(projections, ax, symmetric=True):
    all_proj = np.concatenate(projections, axis=0)

    if symmetric:
        max_abs = np.max(np.abs(all_proj)) * 1.05
        ax.set_xlim(-max_abs, max_abs)
        ax.set_ylim(-max_abs, max_abs)
        return -max_abs, max_abs, -max_abs, max_abs

    else:
        x_min, x_max = all_proj[:, 0].min(), all_proj[:, 0].max()
        y_min, y_max = all_proj[:, 1].min(), all_proj[:, 1].max()

        # Padding
        padding_x = (x_max - x_min) * 0.05
        padding_y = (y_max - y_min) * 0.05

        x_min -= padding_x
        x_max += padding_x
        y_min -= padding_y
        y_max += padding_y

        # Lengths and center
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range)

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2

        # Symmetrize around center with equal aspect ratio
        ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
        ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

        return x_center - max_range / 2, x_center + max_range / 2, y_center - max_range / 2, y_center + max_range / 2


def _interpolate_projections(projections, steps_per_transition):
    interpolated = []
    for a, b in zip(projections[:-1], projections[1:]):
        for alpha_step in np.linspace(0, 1, steps_per_transition, endpoint=False):
            interp = (1 - alpha_step) * a + alpha_step * b
            interpolated.append(interp)
    interpolated.append(projections[-1])
    return np.array(interpolated)

def _prepare_cifar100_plot_config(class_names, cmap='tab20'):
    from vision_classification import get_cifar100_coarse_to_fine_labels, get_cifar100_fine_to_coarse_labels

    coarse_to_fine = get_cifar100_coarse_to_fine_labels()
    fine_to_coarse = get_cifar100_fine_to_coarse_labels()
    coarse_names = list(coarse_to_fine.keys())
    cmap = plt.colormaps.get_cmap(cmap).resampled(len(coarse_names))
    coarse_color_map = {coarse: cmap(i) for i, coarse in enumerate(coarse_names)}
    fine_name_to_index = {name: i for i, name in enumerate(class_names)}

    plot_config = {}
    for coarse in coarse_names:
        fine_list = coarse_to_fine[coarse]
        color = coarse_color_map[coarse]
        for j, fine in enumerate(fine_list):
            fine_idx = fine_name_to_index[fine]
            marker = marker_styles[j % len(marker_styles)]
            plot_config[fine_idx] = {'color': color, 'marker': marker, 'coarse': coarse}
    return plot_config, fine_to_coarse

def _compute_axis_limits(projections_list, shared=True):
    if shared:
        all_proj = np.concatenate([np.concatenate(p) for p in projections_list])
        max_val = np.max(np.abs(all_proj))
        return [(-max_val, max_val)] * len(projections_list)
    else:
        limits = []
        for p in projections_list:
            max_val = np.max(np.abs(np.concatenate(p)))
            limits.append((-max_val, max_val))
        return limits

def _create_cifar100_legend(ax, fine_index_to_plot_config, unique_labels, class_names, fine_to_coarse):
    from collections import defaultdict
    import matplotlib.lines as mlines

    legend_entries = defaultdict(list)
    for fine_idx in unique_labels:
        coarse = fine_to_coarse[class_names[fine_idx]]
        label = class_names[fine_idx]
        marker = fine_index_to_plot_config[fine_idx]['marker']
        color = fine_index_to_plot_config[fine_idx]['color']
        handle = mlines.Line2D([], [], color=color, marker=marker, linestyle='None', label=label)
        legend_entries[coarse].append(handle)

    legend_items = []
    for coarse, handles in legend_entries.items():
        title_handle = mlines.Line2D([], [], color='none', label=f"[{coarse}]", linestyle='None')
        legend_items.append(title_handle)
        legend_items.extend(handles)

    ax.legend(handles=legend_items, bbox_to_anchor=(1.05, 1), loc='upper left',
              title="Fine Classes by Coarse Group")


def adjust_visualization_speed(projections, embedding_drifts, drift_key):
    """
    Adjusts the visualization movement speed iteratively to align with the maximum movement of a specified drift.

    Args:
        projections (list of np.ndarray): Low-dimensional projections.
        embedding_drifts (dict of np.ndarray): High-dimensional embedding drift values.
        target_drift_key (str): Key in embedding_drifts to use as the reference for speed adjustment.

    Returns:
        list of np.ndarray: Adjusted projections with aligned speed.
    """
    # Extract the target drift
    target_drift = np.asarray(embedding_drifts[drift_key]).flatten()
    scaling_difference = np.median(
        _calculate_embedding_drift(projections)[drift_key][drift_key - 1:] / embedding_drifts[drift_key][
                                                                             drift_key - 1:])
    print(f"Scaling difference: {scaling_difference}")

    # Apply scaling iteratively
    adjusted_projections = projections[0:drift_key]  # Start with the first projection as-is
    changes = 0
    for i in range(drift_key, len(projections)):
        # Calculate drift for this step
        current_drift = projections[i] - adjusted_projections[-1]
        vis_drift_step = np.linalg.norm(current_drift, axis=1).mean()
        target_drift_step = np.abs(target_drift[i - 1])  # Reference target drift for this step

        # Determine scaling factor
        if vis_drift_step == 0 or vis_drift_step < target_drift_step * scaling_difference:
            scaling_factor = 1.0
        else:
            scaling_factor = target_drift_step / vis_drift_step * scaling_difference
            changes += 1
            # print(f"{i}: {vis_drift_step} > {target_drift_step}")
            # print(scaling_factor)

        # Apply scaling and update
        adjusted_step = adjusted_projections[-1] + current_drift * scaling_factor
        adjusted_projections.append(adjusted_step)

    print(f"{changes / len(adjusted_projections)}% changes ({changes})")
    return adjusted_projections


def filter_classes(projections, labels, selected_classes):
    projections_filtered = []
    labels_filtered = []

    for proj, lbl in zip(projections, labels):
        mask = np.isin(lbl, selected_classes)
        projections_filtered.append(proj[mask])
        labels_filtered.append(lbl[mask])

    return projections_filtered, labels_filtered


def show_cifar100_legend(dot_size=6, figsize=(8, 6), ncol=4, cmap='tab20'):
    from vision_classification import get_text_labels
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    from collections import defaultdict

    class_names = get_text_labels("cifar100")
    fine_index_to_plot_config, fine_to_coarse = _prepare_cifar100_plot_config(class_names, cmap)

    # Group fine indices by their coarse group.
    legend_entries = defaultdict(list)
    for fine_idx, config in fine_index_to_plot_config.items():
        coarse = config["coarse"]
        label = f"[{fine_idx}] {class_names[fine_idx]}"
        legend_entries[coarse].append((fine_idx, label, config))

    # Build legend handles sorted by coarse groups.
    legend_items = []
    for coarse in sorted(legend_entries.keys()):
        title_handle = mlines.Line2D([], [], color="none", label=f"{coarse.upper()}", linestyle="None")
        legend_items.append(title_handle)
        for _, label, config in sorted(legend_entries[coarse], key=lambda x: x[0]):
            handle = mlines.Line2D(
                [], [],
                color=config["color"],
                marker=config["marker"],
                linestyle="None",
                markersize=dot_size,
                label=label
            )
            legend_items.append(handle)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.legend(
        handles=legend_items,
        loc="center",
        frameon=False,
        title="CIFAR-100 Classes",
        ncol=ncol,
        columnspacing=1.5,
        handletextpad=0.5,
        borderaxespad=0.5,
        fontsize="x-small",
        title_fontsize="medium"
    )
    plt.tight_layout()
    plt.show()