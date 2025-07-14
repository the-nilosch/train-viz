import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm.notebook import tqdm

import plots

marker_styles = ['o', 'p', '^', 'X', 'D', 'P', 'v', '<', '>', '*', "s"]


def visualization_drift_vs_embedding_drift(projections, embedding_drifts, verbose=True, embeddings=False,
                                           figsize=(10, 3), on_ax=None):
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
        figsize (tuple): Size of the figure

    Returns:
        float: Mean Pearson correlation between visualization drift and embedding drift
            across all data series.
        array: In special Case it's a 1D array with series correlations
    """
    if embeddings:
        embedding_drifts = calculate_embedding_drift(embedding_drifts)
    else:
        embedding_drifts = embedding_drifts.copy()

    # Use the existing function to calculate visualization drift
    visualization_drifts = calculate_embedding_drift(projections)

    # Ensure embedding_drifts and visualization_drifts have the same length
    assert len(visualization_drifts) == len(embedding_drifts), \
        f"Mismatch in drift lengths. Vis: {len(visualization_drifts)} vs Emb: {len(embedding_drifts)}"

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
        assert len(emb_drift) == len(vis_drift), f"Mismatch in Length {len(emb_drift)} vs {len(vis_drift)}"
        correlation, _ = pearsonr(emb_drift, vis_drift)
        correlations.append(correlation)
        if verbose:
            print(f"Series {i} - Correlation: {correlation:.4f}")

    # Calculate and print mean correlation
    mean_correlation = np.mean(correlations)

    if on_ax is not None:
        # Part of larger Plot
        plots.plot_embedding_drift(on_ax, visualization_drifts, title="Visualization Drift")
        return correlations

    if verbose:
        print(f"Mean Correlation: {mean_correlation:.4f}")

        fig, axs = plt.subplots(1, 2, figsize=figsize)
        plots.plot_embedding_drift(axs[0], visualization_drifts, title="Visualization Drift")
        plots.plot_embedding_drift(axs[1], embedding_drifts)

        plt.legend()
        plt.show()

    return mean_correlation


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
        embeddings_list,
        method='tsne',
        pca_fit_basis='first',
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
        reducer = PCA(n_components=out_dim)
        reducer.fit(basis_data)
        for i in tqdm(range(max_frames), desc="PCA frames"):
            projections.append(reducer.transform(embeddings_list[i]))

    elif method == 'tsne':
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
