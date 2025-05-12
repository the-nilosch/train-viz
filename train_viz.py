import math
from collections import deque
from pprint import pprint

import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from IPython.display import clear_output
import logging
import ipywidgets as widgets
from IPython.display import display

def train_model_with_embedding_tracking(
    model, train_loader, test_loader, subset_loader, device,
    epochs=10, learning_rate=0.001, embedding_records_per_epoch=10,
    average_window_size=10, track_gradients=True, track_embedding_drift=False,
    track_cosine_similarity=False
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    # Initialize lists for performance tracking
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Initialize lists for embedding snapshot
    num_batches = len(train_loader)
    embedding_batch_interval = math.ceil(num_batches / embedding_records_per_epoch)
    print(f"{num_batches} Batches, {embedding_records_per_epoch} Records per Epoch")
    print("Resulting Batch interval:", embedding_batch_interval)
    embedding_snapshots = []
    embedding_snapshot_labels = []
    embedding_indices = []
    embedding_counter = 0

    # Initialize lists for gradient tracking
    gradient_norms, max_gradients, grad_param_ratios = [], [], []
    batch_indices = []
    batch_counter = 0 # will track absolute batch index for x-axis

    # Logging setup
    logging.basicConfig(level=logging.INFO, force=True)
    log_history = []

    # Visualization setup
    backend = matplotlib.get_backend().lower()
    if 'widget' in backend:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()
    else:
        fig = ax1 = ax2 = None  # placeholder

    for epoch in range(epochs):
        model.train()
        epoch_train_loss, correct_train, total_train = 0, 0, 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            # optimizer.step()

            # Gradient Tracking
            if track_gradients:
                grad_norm, max_grad, grad_ratio = _track_gradients(model)
                gradient_norms.append(grad_norm)
                max_gradients.append(max_grad)
                grad_param_ratios.append(grad_ratio)
            batch_indices.append(batch_counter)
            batch_counter += 1

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

        # Live plot update
        fig, axs = _live_plot_update(track_gradients, track_embedding_drift, track_cosine_similarity)
        _plot_loss_accuracy(axs[0], epoch, epochs, train_losses, val_losses, train_accuracies, val_accuracies)

        pos = 1
        if track_gradients:
            _plot_gradients(axs[pos], batch_indices, gradient_norms, max_gradients, grad_param_ratios, average_window_size)
            pos += 1
        if track_embedding_drift:
            _plot_embedding_drift(axs[pos], embedding_drifts)
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
    }

def _live_plot_update(track_gradients=False, track_embedding_drift=False, track_cosine_similarity=False, ncols=2):
    clear_output(wait=True)
    num_figures = 1 + int(track_gradients) + int(track_embedding_drift) + int(track_cosine_similarity)
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

def _calculate_embedding_drift(embedding_snapshots, max_skip=5):
    """
    Calculate embedding drift based on the snapshots.
    Drift is calculated as the mean Euclidean distance between snapshots.
    """
    drifts = {i: [] for i in range(1, max_skip + 1)}

    # Iterate over all snapshots
    for i in range(1, len(embedding_snapshots)):
        current_snapshot = embedding_snapshots[i]

        # Compare with previous snapshots
        for skip in range(1, max_skip + 1):
            if i - skip >= 0:
                previous_snapshot = embedding_snapshots[i - skip]

                # Calculate Euclidean distance
                drift = np.linalg.norm(current_snapshot - previous_snapshot, axis=1).mean()
                drifts[skip].append(drift)
            else:
                # Not enough snapshots to calculate this skip level
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
    ax.set_xlim(1, epochs)  # Fixed x-axis
    ax.set_ylim(0, max(val_losses + train_losses))  # Fixed range for loss
    ax.set_xticks(list(range(1, epochs + 1)))  # Integer ticks only

    ax2 = ax.twinx()
    ax2.plot(epochs_range, train_accuracies, 'g--', label='Train Acc', alpha=0.7)
    ax2.plot(epochs_range, val_accuracies, 'orange', linestyle='--', label='Val Acc', alpha=0.7)
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(min(train_accuracies + val_accuracies) * 0.9, 1.0)
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
    ax.legend(loc='upper left')

    ax2 = ax.twinx()
    ax2.plot(batch_indices, gradient_norms, 'blue', linestyle='--', label='Gradient Norm', alpha=0.3)
    if len(batch_indices) >= window_size:
        ax2.plot(avg_indices, _moving_average(gradient_norms, window_size), color='blue', linestyle='-',
                 label='Gradient Norm (Avg)', alpha=0.8, linewidth=1.5)
    ax2.set_ylabel('Gradient Norm')
    ax2.legend(loc='upper right')


from scipy.stats import pearsonr


def visualization_drift_vs_embedding_drift(projections, embedding_drifts):
    """
    Visualizes the drift of embeddings and calculates the correlation
    between visualization drift and embedding drift for each data series.

    Args:
        projections (list of np.ndarray): Low-dimensional projections (e.g., t-SNE or UMAP).
        embedding_drifts (list of np.ndarray): High-dimensional embedding drift values, one for each data series.

    Returns:
        float: Mean correlation between visualization drift and embedding drift across all data series.
    """
    embedding_drifts = embedding_drifts.copy()
    # Use the existing function to calculate visualization drift
    visualization_drifts = _calculate_embedding_drift(projections)

    # Ensure embedding_drifts and visualization_drifts have the same length
    assert len(embedding_drifts) == len(visualization_drifts), "Mismatch in drift lengths."

    # Calculate correlation per data series and store in a list
    correlations = []
    for i in range(1, len(embedding_drifts)):
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
        print(f"Series {i} - Correlation: {correlation:.4f}")

    # Calculate and print mean correlation
    mean_correlation = np.mean(correlations)
    print(f"Mean Correlation: {mean_correlation:.4f}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    _plot_embedding_drift(axs[0], visualization_drifts, title="Visualization Drift")
    _plot_embedding_drift(axs[1], embedding_drifts)

    plt.legend()
    plt.show()

    return mean_correlation

def _plot_embedding_drift(ax, embedding_drifts, title="Embedding Drift"):
    """Plots embedding drift."""
    colors = ['green', 'blue', 'orange', 'red', 'purple']
    labels = ['Drift 1', 'Drift 2', 'Drift 3', 'Drift 4', 'Drift 5']

    for skip, color, label in zip(range(1, 6), colors, labels):
        drift_data = embedding_drifts[skip]
        if len(drift_data) > 0:
            indices = range(1, len(drift_data) + 1)
            ax.plot(indices, drift_data, color=color, label=label, alpha=0.7)

    ax.set_title(title)
    ax.set_ylabel("Drift Distance")
    ax.set_xlabel("Snapshot Index")
    ax.legend(loc='upper left')

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
):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    import numpy as np

    assert method in ['tsne', 'pca', 'umap']
    assert pca_fit_basis in ['first', 'last', 'all', 'last_visualized']

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
    else:
        raise ValueError(f"Invalid pca_fit_basis: {pca_fit_basis}")

    if method == 'pca':
        reducer = PCA(n_components=2)
        reducer.fit(basis_data)
        for i in range(max_frames):
            projections.append(reducer.transform(embeddings_list[i]))

    elif method == 'tsne':
        tsne = TSNE(n_components=2, init=tsne_init, perplexity=tsne_perplexity, random_state=random_state, metric=metric)
        tsne.fit(basis_data)
        if reverse_computation:
            projections = [None] * max_frames
            projections[-1] = tsne.fit_transform(embeddings_list[max_frames - 1])
            for i in range(max_frames - 2, -1, -1):
                tsne = TSNE(n_components=2, init=projections[i + 1], perplexity=tsne_perplexity, random_state=random_state, metric=metric)
                projections[i] = tsne.fit_transform(embeddings_list[i])
        else:
            projections.append(tsne.fit_transform(embeddings_list[0]))
            for i in range(1, max_frames):
                tsne = TSNE(n_components=2, init=projections[-1], perplexity=tsne_perplexity, random_state=random_state, metric=metric)
                projections.append(tsne.fit_transform(embeddings_list[i]))

    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, metric=metric)
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

#     elif method == 'umap':
#         init_2d = 'pca'
#         for i in range(max_frames):
#             prev_emb = embeddings_list[i - 1] if i > 0 else embeddings_list[i]
#             next_emb = embeddings_list[i + 1] if i < len(embeddings_list) - 1 else embeddings_list[i]
#             curr_emb = embeddings_list[i]
#             fit_data = np.concatenate([prev_emb, curr_emb, next_emb], axis=0)
#
#             reducer = umap.UMAP(n_components=2, init=init_2d)
#             reducer.fit(fit_data)
#             projection = reducer.transform(curr_emb)
#
#             projections.append(projection)
#             init_2d = projection  # use current as init for next

    return projections


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
        cmap='tab10',
        dot_size=5,
        alpha=0.5,
        interpolate=False,
        steps_per_transition=10
    ):
    projections = np.array(projections)

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

    projections = projections_interp

    # Setup figure
    fig, ax = plt.subplots(figsize=figsize)

    # Axis limits
    all_proj = np.concatenate(projections, axis=0)
    max_abs = np.max(np.abs(all_proj))
    ax.set_xlim(-max_abs, max_abs)
    ax.set_ylim(-max_abs, max_abs)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Initial scatter
    scatter = ax.scatter(projections[0][:, 0], projections[0][:, 1],
                         c=labels[0], cmap=cmap, s=dot_size, alpha=alpha)

    # Slider and update
    def update(frame_idx):
        scatter.set_offsets(projections[frame_idx])
        scatter.set_array(np.array(labels[0]))
        fig.canvas.draw_idle()

    slider = widgets.Play(min=0, max=len(projections)-1, step=1)
    slider_control = widgets.IntSlider(min=0, max=len(projections)-1, step=1)
    widgets.jslink((slider, 'value'), (slider_control, 'value'))

    out = widgets.interactive_output(update, {'frame_idx': slider_control})
    display(widgets.VBox([widgets.HBox([slider, slider_control]), out]))


def show_multiple_projections_with_slider(
    projections_list,
    labels,
    titles=None,
    figsize_per_plot=(5, 5),
    cmap='tab10',
    dot_size=5,
    alpha=0.6,
    interpolate=False,
    steps_per_transition=10,
    shared_axes=True
):
    num_views = len(projections_list)
    projections_list = [np.array(p) for p in projections_list]

    # Interpolation
    def interpolate_projections(projs):
        if not interpolate:
            return projs
        projs_interp = []
        for a, b in zip(projs[:-1], projs[1:]):
            for alpha_step in np.linspace(0, 1, steps_per_transition, endpoint=False):
                interp = (1 - alpha_step) * a + alpha_step * b
                projs_interp.append(interp)
        projs_interp.append(projs[-1])
        return np.array(projs_interp)

    projections_list = [interpolate_projections(p) for p in projections_list]

    # Verify all have the same number of frames
    n_frames = len(projections_list[0])
    for p in projections_list:
        assert len(p) == n_frames, "All projection sets must have the same number of frames"

    # === Layout calculation ===
    if num_views <= 4:
        nrows, ncols = 1, num_views
    else:
        ncols = math.ceil(math.sqrt(num_views))
        nrows = math.ceil(num_views / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))
    axes = np.array(axes).reshape(-1)  # Flatten in case it's a 2D grid

    # Trim excess axes if more than needed
    for ax in axes[num_views:]:
        fig.delaxes(ax)

    axes = axes[:num_views]

    # Precompute axis limits
    if shared_axes:
        all_proj = np.concatenate([np.concatenate(p) for p in projections_list])
        global_max = np.max(np.abs(all_proj))
        axis_limits = [(-global_max, global_max)] * num_views
    else:
        axis_limits = []
        for p in projections_list:
            max_val = np.max(np.abs(np.concatenate(p)))
            axis_limits.append((-max_val, max_val))

    # Create initial scatter plots
    scatters = []
    for i, ax in enumerate(axes):
        xlim, ylim = axis_limits[i], axis_limits[i]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        title = titles[i] if titles and i < len(titles) else f"View {i+1}"
        ax.set_title(title)

        sc = ax.scatter(projections_list[i][0][:, 0], projections_list[i][0][:, 1],
                        c=labels[0], cmap=cmap, s=dot_size, alpha=alpha)
        scatters.append(sc)

    # Update all subplots
    def update(frame_idx):
        for i in range(num_views):
            scatters[i].set_offsets(projections_list[i][frame_idx])
            scatters[i].set_array(np.array(labels[0]))
        fig.canvas.draw_idle()

    # Interactive slider + play
    slider = widgets.Play(min=0, max=n_frames - 1, step=1)
    slider_control = widgets.IntSlider(min=0, max=n_frames - 1, step=1)
    widgets.jslink((slider, 'value'), (slider_control, 'value'))

    out = widgets.interactive_output(update, {'frame_idx': slider_control})
    display(widgets.VBox([widgets.HBox([slider, slider_control]), out]))
