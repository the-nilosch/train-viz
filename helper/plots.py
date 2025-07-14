import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import ipywidgets as widgets
from IPython.display import display

marker_styles = ['o', 'p', '^', 'X', 'D', 'P', 'v', '<', '>', '*', "s"]


def plot_loss_accuracy(ax, epoch, epochs, train_losses, val_losses, train_accuracies, val_accuracies):
    """Plots loss and accuracy over epochs."""
    ax.set_title("Loss & Accuracy per Epoch")
    epochs_range = range(1, epoch + 2)
    ax.plot(epochs_range, train_losses, 'b-', label='Train Loss', alpha=0.7)
    ax.plot(epochs_range, val_losses, 'r-', label='Val Loss', alpha=0.7)
    ax.set_ylabel('Loss')
    ax.legend(loc='upper left')
    # ax.set_xlim(1, epochs)  # Fixed x-axis
    ax.set_ylim(0, max(val_losses + train_losses))  # Fixed range for loss
    # ax.set_xticks(list(range(1, epochs + 1)))  # Integer ticks only

    ax2 = ax.twinx()
    ax2.plot(epochs_range, train_accuracies, 'g--', label='Train Acc', alpha=0.7)
    ax2.plot(epochs_range, val_accuracies, 'orange', linestyle='--', label='Val Acc', alpha=0.7)
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(min(min(train_accuracies), min(val_accuracies)) * 0.9, 1.0)
    ax2.legend(loc='upper right')


def plot_gradients(ax, batch_indices, gradient_norms, max_gradients, grad_param_ratios, window_size):
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


def plot_scheduled_lr(ax, scheduler_history):
    ax.plot(scheduler_history, label='Learning Rate')
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")


def plot_pca(ax, embedding_snapshots, embedding_snapshot_labels, embedding_records_per_epoch, out_dim=2,
              num_classes=10, size=10):
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

def plot_embedding_drift(ax, embedding_drifts, title="Embedding Drift", max_multiply=1.1):
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
    from helper.vision_classification import get_text_labels
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
        dataset=None,
        axes=None,
        fig=None
):
    from helper.vision_classification import get_text_labels
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

    if axes is None:
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))
        axes = np.array(axes).reshape(-1)

        for ax in axes[num_views:]:
            fig.delaxes(ax)
        axes = axes[:num_views]

    assert fig is not None, "When passing axes, it also needs a figure"

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

        title = titles[i] if titles and i < len(titles) else f"View {i + 1}"
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
    from helper.vision_classification import get_cifar100_coarse_to_fine_labels, get_cifar100_fine_to_coarse_labels

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
    from helper.visualization import calculate_embedding_drift

    target_drift = np.asarray(embedding_drifts[drift_key]).flatten()
    scaling_difference = np.median(
        calculate_embedding_drift(projections)[drift_key][drift_key - 1:] / embedding_drifts[drift_key][
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
    from helper.vision_classification import get_text_labels
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


def show_with_slider_3d(
        projections,
        labels,
        dot_size=5,
        alpha=0.6,
        interpolate=False,
        steps_per_transition=10,
        dataset=None,
        show_legend=False  # placeholder for symmetry
):
    from helper.vision_classification import get_text_labels
    class_names = range(0, 100) if dataset is None else get_text_labels(dataset)
    projections = np.array(projections)
    projections = _interpolate_projections(projections, steps_per_transition) if interpolate else projections

    if dataset == "cifar100":
        fine_index_to_plot_config, fine_to_coarse = _prepare_cifar100_plot_config(class_names)

    unique_labels = np.unique(np.concatenate(labels))
    label_frame = labels[0]  # assumed constant

    # Set up 3D plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    all_proj = np.concatenate(projections, axis=0)
    max_abs = np.max(np.abs(all_proj))
    ax.set_xlim3d(-max_abs, max_abs)
    ax.set_ylim3d(-max_abs, max_abs)
    ax.set_zlim3d(-max_abs, max_abs)
    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.set_zticks([])

    # Create initial scatter per fine class
    scatter_dict = {}
    if dataset == "cifar100":
        for fine_idx in unique_labels:
            idxs = label_frame == fine_idx
            config = fine_index_to_plot_config.get(fine_idx, {})
            color = config.get('color', 'gray')
            marker = config.get('marker', 'o')
            sc = ax.scatter(projections[0][idxs, 0],
                            projections[0][idxs, 1],
                            projections[0][idxs, 2],
                            c=[color], marker=marker,
                            alpha=alpha, edgecolors='none', s=dot_size * 2)
            scatter_dict[fine_idx] = sc
    else:
        sc = ax.scatter(projections[0][:, 0],
                        projections[0][:, 1],
                        projections[0][:, 2],
                        c=label_frame, cmap='tab10', alpha=alpha, s=dot_size)

    # Update function
    def update(frame_idx, azim_angle, elev_angle, auto_rotate):
        if dataset == "cifar100":
            for fine_idx in unique_labels:
                idxs = label_frame == fine_idx
                scatter_dict[fine_idx]._offsets3d = (
                    projections[frame_idx][idxs, 0],
                    projections[frame_idx][idxs, 1],
                    projections[frame_idx][idxs, 2]
                )
        else:
            sc._offsets3d = (
                projections[frame_idx][:, 0],
                projections[frame_idx][:, 1],
                projections[frame_idx][:, 2]
            )
            sc.set_array(np.array(label_frame))

        # Only update view if auto-rotate is enabled
        if auto_rotate:
            ax.view_init(elev=elev_angle, azim=azim_angle)

        fig.canvas.draw_idle()

    def on_azim_change(change):
        toggle_auto_rotate.value = True

    # Sliders
    slider_frame = widgets.IntSlider(min=0, max=len(projections) - 1, step=1, description="Frame")
    slider_azim = widgets.IntSlider(min=0, max=360, step=1, description="Rotation")
    slider_azim.observe(on_azim_change, names='value')
    toggle_auto_rotate = widgets.Checkbox(value=True, description="Auto-Rotate")

    # Play controls
    play_frame = widgets.Play(interval=150 / steps_per_transition, value=0, min=0, max=len(projections) - 1, step=1,
                              description="▶")
    play_azim = widgets.Play(interval=100, value=0, min=0, max=360, step=2, description="↻")
    play_azim.observe(on_azim_change, names='value')
    play_azim.loop = True

    slider_elev = widgets.IntSlider(min=-90, max=90, step=1, value=30, description="Tilt")
    play_elev = widgets.Play(interval=100, min=-90, max=90, step=1, description="↕")
    play_elev.loop = True
    widgets.jslink((play_elev, 'value'), (slider_elev, 'value'))

    # Link sliders to play widgets
    widgets.jslink((play_elev, 'value'), (slider_elev, 'value'))
    widgets.jslink((play_frame, 'value'), (slider_frame, 'value'))
    widgets.jslink((play_azim, 'value'), (slider_azim, 'value'))

    out = widgets.interactive_output(update, {
        'frame_idx': slider_frame,
        'azim_angle': slider_azim,
        'elev_angle': slider_elev,
        'auto_rotate': toggle_auto_rotate
    })

    display(widgets.VBox([
        widgets.HBox([play_frame, slider_frame]),
        widgets.HBox([play_azim, slider_azim]),
        widgets.HBox([play_elev, slider_elev]),
        toggle_auto_rotate,
        out
    ]))

def _moving_average(data, window_size):
    """Calculate the moving average with a specified window size."""
    if len(data) < window_size:
        return data  # Not enough data points to calculate the average
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')