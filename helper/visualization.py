import numpy as np
import matplotlib.pyplot as plt
from pyparsing import results
from scipy.stats import pearsonr
from tqdm.notebook import tqdm

import helper.plots as plots
from helper.data_manager import load_training_data

marker_styles = ['o', 'p', '^', 'X', 'D', 'P', 'v', '<', '>', '*', "s"]

class Run:
    def __init__(self, run_id: str, dataset: str):
        self.run_id = run_id
        self.dataset = dataset

        self.results = self.load()
        self.embedding_drifts = self.results['embedding_drifts']
        self.embeddings = self.results["subset_embeddings"]
        self.labels = self.results["subset_labels"]

    def load(self):
        loaded_results = load_training_data(self.run_id)
        loaded_results["embedding_drifts"] = {int(k): loaded_results["embedding_drifts"][k] for k in
                                       sorted(loaded_results["embedding_drifts"].keys(), key=int)}
        return loaded_results

    def plot_training_records(self):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        epochs = len(self.results["train_losses"])

        plots.plot_loss_accuracy(axs[0][0], epochs - 1, epochs, self.results["train_losses"], self.results["val_losses"],
                           self.results["train_accuracies"], self.results["val_accuracies"])
        plots.plot_gradients(axs[0][1], range(0, len(self.results["gradient_norms"])), self.results["gradient_norms"],
                       self.results["max_gradients"], self.results["grad_param_ratios"], 20)
        if "scheduler_history" in self.results.keys():
            plots.plot_scheduled_lr(axs[1][0], self.results["scheduler_history"])
        plots.plot_embedding_drift(axs[1][1], self.results["embedding_drifts"])

    def plot_embedding_drifts(self, doubled_lines = True):
        embedding_drifts = self.embedding_drifts.copy()
        fig, axs = plt.subplots(1, 1, figsize=(10, 4))

        # Plot 2x Drifts
        if doubled_lines:
            axs.plot(range(1, len(embedding_drifts[1]) + 1), np.array(embedding_drifts[1]) * 2, color="green",
                     label="2x Drift 1", alpha=0.3)
            axs.plot(range(1, len(embedding_drifts[2]) + 1), np.array(embedding_drifts[2]) * 2, color="blue",
                     label="2x Drift 2", alpha=0.3)
            axs.plot(range(1, len(embedding_drifts[4]) + 1), np.array(embedding_drifts[4]) * 2, color="orange",
                     label="2x Drift 2", alpha=0.3)
            axs.plot(range(1, len(embedding_drifts[8]) + 1), np.array(embedding_drifts[8]) * 2, color="red",
                     label="2x Drift 2", alpha=0.3)

        plots.plot_embedding_drift(axs, embedding_drifts, max_multiply=1.5)

        plt.legend()
        plt.show()

    def print_info(self):
        print(f"Sequene of {len(self.embeddings)} snapshots")
        print(f"One Snapshot has (n, dim) = {self.embeddings[0].shape}")

    def subsample(self, snapshot_step: int = 2, point_step: int = 2):
        """
        Subsample the stored embeddings in place and recompute embedding drifts.

        Parameters
        ----------
        snapshot_step : int, optional (default=2)
        point_step : int, optional (default=2)
        """

        # subsample the time axis (snapshots)
        self.embeddings = self.embeddings[::snapshot_step]

        # subsample within each embedding (points)
        self.embeddings = [emb[::point_step, :] for emb in self.embeddings]

        # recompute embedding drift on the subsampled embeddings
        self.embedding_drifts = calculate_embedding_drift(self.embeddings)

        self.labels = self.labels[::snapshot_step]
        self.labels = [labels[::point_step] for labels in self.labels]

        return self

    def confusion_matrix(
            self,
            figsize=(5, 5),
            cmap='Blues',
            annotate=False,
            interval=500
    ):
        """
        Visualize a sequence of confusion matrices with a Play+Slider widget,
        updating only the image data and text annotations on the same canvas.
        """
        plots.show_confusion_slider(
            self.results['val_confusion_matrices'],
            self.dataset,
            figsize=figsize,
            cmap='Blues',
            annotate=annotate,
            interval=interval
        )


class Animation:
    def __init__(self, projections: list, title: str, run: Run):
        self.projections = projections
        self.title = title
        # From Run
        self.run = run
        self.run_id = run.run_id
        self.labels = run.labels
        self.embedding_drifts = run.embedding_drifts.copy()

    def copy(self):
        return Animation(self.projections.copy(), self.title, self.run)

    def plot(self, figsize=(5, 5), dot_size=5, alpha=0.5, interpolate=False, steps_per_transition=10, symmetric=False):
        plots.show_with_slider(
            self.projections,
            self.labels,
            figsize=figsize,
            dot_size=dot_size,
            alpha=alpha,
            interpolate=interpolate,
            steps_per_transition=steps_per_transition,
            dataset=self.run.dataset,
            show_legend=False if self.run.dataset == "cifar100" else True,
            symmetric=symmetric
        )

    def save_as_gif(
            self,
            frame_interval=50,
            figsize=(4, 4),
            dot_size=5,
            alpha=0.6,
            cmap='tab10',
            axis_lim=None,
            interpolate=True,
            steps_per_transition=1,
    ):
        print("Generating plot...")
        ani = animate_projections(
            self.projections,
            self.labels,
            frame_interval=frame_interval,
            interpolate=interpolate,
            steps_per_transition=steps_per_transition,
            figsize=figsize,
            dot_size=dot_size,
            alpha=alpha,
            cmap=cmap,
            axis_lim=axis_lim
        )

        print("Saving file...")
        filename = f"plots/animations/{self.run_id}_{self.title}.gif"

        ani.save(filename, writer='pillow', dpi=150)
        plt.close(ani._fig)

        print(filename)

    def evaluate(self, verbose=True, figsize=(10, 3)):
        return visualization_drift_vs_embedding_drift(
            self.projections,
            self.embedding_drifts,
            verbose=verbose,
            figsize=figsize,
        )

    def denoise(self, blend=0.9, window_size=15, mode='window', do_projections=True, do_embedding_drift=True):
        copy = self.copy()

        if do_projections:
            copy.projections = denoise_projections(
                self.projections.copy(),
                window_size=window_size,
                blend=blend,
                mode=mode)
            copy.title += f" denoised"

        if do_embedding_drift:
            copy.embedding_drifts = calculate_embedding_drift(
                denoise_projections(
                    self.run.embeddings,
                    window_size=window_size,
                    blend=blend,
                    mode=mode
                )
            )

        return copy







def visualization_drift_vs_embedding_drift(projections, embedding_drifts, verbose=True, embeddings=False,
                                           figsize=(10, 3), on_ax=None):
    """
    Computes the composite similarity score between visualization and embedding drift
    across all skip levels, and optionally visualizes the results.
    """
    if embeddings:
        embedding_drifts = calculate_embedding_drift(embedding_drifts)
    else:
        embedding_drifts = embedding_drifts.copy()

    visualization_drifts = calculate_embedding_drift(projections)

    assert len(visualization_drifts) == len(embedding_drifts), \
        f"Mismatch in drift lengths. Vis: {len(visualization_drifts)} vs Emb: {len(embedding_drifts)}"

    # Compute similarity
    mean_similarity, similarity_scores = compute_drift_similarity_score(
        embedding_drifts, visualization_drifts, verbose=verbose
    )

    if on_ax is not None:
        plots.plot_embedding_drift(on_ax, visualization_drifts, title="Visualization Drift")
        return similarity_scores

    if verbose:
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        plots.plot_embedding_drift(axs[0], visualization_drifts, title="Visualization Drift")
        plots.plot_embedding_drift(axs[1], embedding_drifts)
        plt.legend()
        plt.show()

    return mean_similarity


def calculate_embedding_drift(embedding_snapshots, max_power=5):
    """
    Calculate embedding drift based on the snapshots.
    Drift is calculated as the mean Euclidean distance between snapshots.
    Uses skip steps as powers of 2 (i.e., 1, 2, 4, 8, ...).
    """
    drifts = {2 ** n: [] for n in range(max_power)}

    # Iterate over all snapshots
    for i in range(1, len(embedding_snapshots)):
        current_snapshot = embedding_snapshots[i]

        # Compare with previous snapshots using 2^n steps
        for n in range(max_power):
            skip = 2 ** n
            if i - skip >= 0:
                previous_snapshot = embedding_snapshots[i - skip]
                drift = np.linalg.norm(current_snapshot - previous_snapshot, axis=1).mean()
                drifts[skip].append(drift)
            else:
                drifts[skip].append(np.nan)

    return drifts

from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt

def compute_drift_similarity_score(embedding_drifts, visualization_drifts, eps=1e-6, verbose=True):
    """
    Computes log-transformed correlation × ratio penalty for each skip length.

    Returns:
        float: mean similarity
        dict: similarity per skip
    """
    similarity_scores = {}

    for k in embedding_drifts.keys():
        emb = np.asarray(embedding_drifts[k])
        vis = np.asarray(visualization_drifts[k])

        mask = ~np.isnan(emb) & ~np.isnan(vis)
        if np.sum(mask) < 3:
            similarity_scores[k] = np.nan
            continue

        emb = emb[mask]
        vis = vis[mask]

        # Log-correlation
        log_emb = np.log(emb + eps)
        log_vis = np.log(vis + eps)
        log_corr, _ = pearsonr(log_emb, log_vis)

        # Ratio penalty with median normalization
        emb_ratio = emb / (np.median(emb) + eps)
        vis_ratio = vis / (np.median(vis) + eps)
        delta = np.abs(emb_ratio - vis_ratio)
        ratio_penalty = np.exp(-np.mean(delta))

        similarity = log_corr * ratio_penalty
        similarity_scores[k] = similarity

        if verbose:
            print(f"Skip {k} — log-corr: {log_corr:.3f}, ratio: {ratio_penalty:.3f}, similarity: {similarity:.3f}")

    mean_similarity = np.nanmean(list(similarity_scores.values()))
    if verbose:
        print(f"\nMean Composite Similarity: {mean_similarity:.4f}")
    return mean_similarity, similarity_scores


def show_projections_and_drift(projections_list, titles, labels, embedding_drifts, embeddings=False, interpolate=False,
                               steps_per_transition=5, figsize_embedding_drift=(6, 4), figsize_visualization=(4, 4),
                               shared_axes=False, dot_size=5, alpha=0.6, dataset=None):
    """
    Combined static figure with:
    - Left: multiple projection snapshots at chosen frame (slider)
    - Right: embedding vs visualization drift plot
    """
    if embeddings:
        embedding_drifts = calculate_embedding_drift(embedding_drifts)  # MAY NOT BE NEEDED

    nrows = len(projections_list)
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 4 * nrows))
    axes_left = axes[:, 0]
    axes_right = axes[:, 1]

    # Show projection snapshots
    plots.show_multiple_projections_with_slider(
        projections_list,
        labels,
        titles=titles,
        figsize_per_plot=figsize_visualization,
        dot_size=dot_size,
        alpha=alpha,
        interpolate=interpolate,
        steps_per_transition=steps_per_transition,
        shared_axes=shared_axes,
        dataset=dataset,
        axes=axes_left,
        fig=fig
    )

    # Calculate Visualization Drifts
    correlations_list = []
    for i, projection in enumerate(projections_list):
        correlations_list.append(
            visualization_drift_vs_embedding_drift(projection,
                                                   embedding_drifts,
                                                   verbose=False,
                                                   embeddings=embeddings,
                                                   figsize=figsize_embedding_drift,
                                                   on_ax=axes_right[i]))

    plt.show()
    correlation_means = [np.mean(corrs) for corrs in correlations_list]
    for i in range(nrows):
        print(f"{titles[i]}: {correlation_means[i]} = {correlations_list[i]}")


def generate_projections(
        run: Run,
        method='pca',
        pca_fit_basis='all',
        max_frames=None,
        reverse_computation=False,
        random_state=42,
        tsne_init='pca',  # 'pca' or 'random'
        tsne_perplexity=30.0,  # often between 5–50
        tsne_update=1,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        metric='euclidean',  # for umap and tsne
        window_size=10,
        out_dim=2  # 2D vs 3D
):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    import numpy as np

    assert method in ['tsne', 'pca', 'umap']
    assert pca_fit_basis in ['first', 'last', 'all', 'last_visualized', 'window', int]

    projections = []
    embeddings_list = run.embeddings.copy()
    max_frames = max_frames or len(embeddings_list)
    title="dummy"

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
        title = f'PCA on window (size {window_size})'
        from scipy.linalg import orthogonal_procrustes

        prev_components = None
        prev_projection = None

        for i in tqdm(range(max_frames), desc="PCA frames"):
            window_start = max(0, i - window_size + 1)
            window_data = np.concatenate(embeddings_list[window_start:i + 1], axis=0)
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
        title = f'PCA on {pca_fit_basis}'
        reducer = PCA(n_components=out_dim)
        reducer.fit(basis_data)
        for i in tqdm(range(max_frames), desc="PCA frames"):
            projections.append(reducer.transform(embeddings_list[i]))

    elif method == 'tsne':
        title = (f't-SNE (init={tsne_init}, '
                f'perplexity={tsne_perplexity}, '
                f'{metric}, '
                f'{"reverse computation, " if reverse_computation else ""}'
                f'{"" if tsne_update == 1 else f"blending={tsne_update},"}'
                 f')')
        print("Initializing t-SNE...")
        tsne = TSNE(n_components=out_dim, init=tsne_init, perplexity=tsne_perplexity, random_state=random_state,
                    metric=metric)
        tsne.fit(basis_data)
        if reverse_computation:
            projections = [None] * max_frames
            projections[-1] = tsne.fit_transform(embeddings_list[max_frames - 1])
            for i in tqdm(
                    range(max_frames - 2, -1, -1),  # start, stop, step
                    desc="t-SNE frames"):
                tsne = TSNE(n_components=out_dim, init=projections[i + 1], perplexity=tsne_perplexity,
                            random_state=random_state, metric=metric)
                new = tsne.fit_transform(embeddings_list[i]) * tsne_update + projections[i + 1] * (1 - tsne_update)
                projections[i] = tsne.fit_transform(new)
        else:
            projections.append(tsne.fit_transform(embeddings_list[0]))
            for i in tqdm(range(1, max_frames), desc="t-SNE frames"):
                tsne = TSNE(n_components=out_dim, init=projections[-1], perplexity=tsne_perplexity,
                            random_state=random_state, metric=metric)
                new = tsne.fit_transform(embeddings_list[i]) * tsne_update + projections[-1] * (1 - tsne_update)
                projections.append(new)

    elif method == 'umap':
        title = (f'UMAP (n={umap_n_neighbors}, dist={umap_min_dist}, {metric}, '
                f'{"reverse computation, " if reverse_computation else ""}'
                 f')')
        reducer = umap.UMAP(n_components=out_dim, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, metric=metric)
        if reverse_computation:
            for i in tqdm(range(max_frames - 1, -1, -1), desc="UMAP frames"):
                reducer.fit(embeddings_list[i])
                projection = reducer.transform(embeddings_list[i])
                projections.insert(0, projection)
        else:
            reducer.fit(basis_data)
            for i in tqdm(range(max_frames), desc="UMAP frames"):
                projection = reducer.transform(embeddings_list[i])
                projections.append(projection)

    return Animation(projections=projections, title=title, run=run)


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
    assert mode in ['exponential', 'window'], "Mode must be 'exponential' or 'window'"

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


def show_animations(
        animations: list[Animation],
        custom_titles=None,
        figsize_per_plot=(4, 4),
        dot_size=5,
        alpha=0.6,
        interpolate=False,
        steps_per_transition=5,
        shared_axes=True,
        with_drift=False,
        add_confusion_matrix=False,
        annotate_confusion_matrix=False
):
    projections_list = [ani.projections for ani in animations]
    titles = custom_titles if custom_titles else [ani.title for ani in animations]

    if not with_drift:
        plots.show_multiple_projections_with_slider(
            projections_list=projections_list,
            labels=animations[0].labels,
            titles=titles,
            figsize_per_plot=figsize_per_plot,
            dot_size=dot_size,
            alpha=alpha,
            interpolate=interpolate,
            steps_per_transition=steps_per_transition,
            shared_axes=shared_axes,
            dataset=animations[0].run.dataset,
            confusion_matrices=animations[0].run.results['val_confusion_matrices'] if add_confusion_matrix else None,
            annotate_conf=annotate_confusion_matrix
        )
        return

    embedding_drifts = [ani.embedding_drifts for ani in animations]
    show_projections_and_drift(
        projections_list = projections_list,
        titles = titles,
        labels=animations[0].labels,
        embedding_drifts= embedding_drifts,
        embeddings = False,
        interpolate=interpolate,
        steps_per_transition=steps_per_transition,
        figsize_embedding_drift=(6, 4),
        figsize_visualization=(4, 4),
        shared_axes=shared_axes,
        dot_size = dot_size,
        alpha = alpha,
    )


def animate_projections(
        projections,
        labels,
        interpolate=False,
        steps_per_transition=10,
        frame_interval=50,
        figsize=(4, 4),
        dot_size=5,
        alpha=0.6,
        cmap='tab10',
        title_base='Embedding Evolution',
        axis_lim=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Interpolate if requested
    projections = plots._interpolate_projections(projections, steps_per_transition) if interpolate else projections

    # Auto axis limit
    if axis_lim is None:
        all_proj = np.concatenate(projections, axis=0)
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
        scatter.set_offsets(projections[frame])
        scatter.set_array(np.array(labels).flatten())
        return scatter,

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(projections),
        interval=frame_interval,
        blit=True
    )

    return ani
