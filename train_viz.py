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
        model,
        train_loader,
        test_loader,
        test_subset_loader,
        device,
        epochs=10,
        learning_rate=0.001,
        embedding_mode='batch',  # 'batch' or 'epoch'
        batch_interval=10  # Only used if mode == 'batch'
):
    logging.basicConfig(level=logging.INFO, force=True)
    log_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    test_subset_embeddings = []
    test_subset_labels = []

    # Place this outside your loop (once)
    backend = matplotlib.get_backend().lower()
    if 'widget' in backend:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()
    else:
        fig = ax1 = ax2 = None  # placeholder

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output, embedding = model(data, return_embedding=True)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Training metrics
            epoch_train_loss += loss.item()
            _, preds = torch.max(output, dim=1)
            correct_train += (preds == target).sum().item()
            total_train += target.size(0)

            # === Embedding tracking during training ===
            if embedding_mode == 'batch' and (batch_idx % batch_interval == 0):
                model.eval()
                with torch.no_grad():
                    batch_embeddings = []
                    batch_labels = []
                    for data_sub, target_sub in test_subset_loader:
                        data_sub, target_sub = data_sub.to(device), target_sub.to(device)
                        _, emb = model(data_sub, return_embedding=True)
                        batch_embeddings.append(emb.cpu().numpy())
                        batch_labels.append(target_sub.cpu().numpy())
                    test_subset_embeddings.append(np.concatenate(batch_embeddings, axis=0))
                    test_subset_labels.append(np.concatenate(batch_labels, axis=0))
                model.train()

        # === Embedding tracking once per epoch ===
        if embedding_mode == 'epoch':
            model.eval()
            with torch.no_grad():
                batch_embeddings = []
                batch_labels = []
                for data_sub, target_sub in test_subset_loader:
                    data_sub, target_sub = data_sub.to(device), target_sub.to(device)
                    _, emb = model(data_sub, return_embedding=True)
                    batch_embeddings.append(emb.cpu().numpy())
                    batch_labels.append(target_sub.cpu().numpy())
                test_subset_embeddings.append(np.concatenate(batch_embeddings, axis=0))
                test_subset_labels.append(np.concatenate(batch_labels, axis=0))

        # === Epoch-wise accuracy ===
        train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # === Validation phase ===
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data, return_embedding=True)
                loss = criterion(output, target)
                epoch_val_loss += loss.item()
                _, preds = torch.max(output, dim=1)
                correct_val += (preds == target).sum().item()
                total_val += target.size(0)

        val_loss = epoch_val_loss / len(test_loader)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # === Live plot update ===
        clear_output(wait=True)
        epochs_range = range(1, epoch + 2)
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs_range, val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_ylim(0, max(val_losses + train_losses))  # Fixed range for loss
        ax1.set_xlim(1, epochs)  # Fixed x-axis
        ax1.set_xticks(list(range(1, epochs + 1)))  # Integer ticks only
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(epochs_range, train_accuracies, 'g--', label='Train Acc')
        ax2.plot(epochs_range, val_accuracies, 'orange', linestyle='--', label='Val Acc')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(min(train_accuracies + val_accuracies), 1.0)  # Fixed range for accuracy
        ax2.legend(loc='upper right')

        plt.title("Training Loss and Accuracy")
        plt.tight_layout()
        plt.show()

        # === Print summary ===
        log_line = (
            f"Epoch [{epoch + 1}/{epochs}] "
            f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )
        log_history.append(log_line)
        print("\n".join(log_history))

    print(
        f"Recorded {len(test_subset_embeddings)} embeddings in {epochs} epochs "
        f"({len(test_subset_embeddings) / epochs:.2f} per epoch)."
    )

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_subset_embeddings': test_subset_embeddings,
        'test_subset_labels': test_subset_labels
    }


def generate_projections(
    embeddings_list,
    method='tsne',
    pca_fit_basis='first',
    max_frames=None
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
    if pca_fit_basis == 'first':
        basis_data = embeddings_list[0]
    elif pca_fit_basis == 'last':
        basis_data = embeddings_list[-1]
    elif pca_fit_basis == 'last_visualized':
        basis_data = embeddings_list[max_frames - 1]
    else:  # 'all'
        basis_data = np.concatenate(embeddings_list, axis=0)

    if method == 'pca':
        reducer = PCA(n_components=2)
        reducer.fit(basis_data)
        for i in range(max_frames):
            projections.append(reducer.transform(embeddings_list[i]))

    elif method == 'tsne':
        tsne = TSNE(n_components=2, init='pca', random_state=42)
        projections.append(tsne.fit_transform(embeddings_list[0]))
        for i in range(1, max_frames):
            tsne = TSNE(n_components=2, init=projections[-1], random_state=42)
            projections.append(tsne.fit_transform(embeddings_list[i]))

    elif method == 'umap':
        reducer = umap.UMAP(n_components=2)
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

    # Then inside your code
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

    # Setup figure with dynamic subplots
    fig, axes = plt.subplots(1, num_views, figsize=(figsize_per_plot[0] * num_views, figsize_per_plot[1]))

    if num_views == 1:
        axes = [axes]

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
