import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm.notebook import tqdm

import helper.plots as plots

marker_styles = ['o', 'p', '^', 'X', 'D', 'P', 'v', '<', '>', '*', "s"]

class Run:
    def __init__(self, run_id: str, dataset: str):
        self.run_id = run_id
        self.dataset = dataset

        self.results = self.load()
        self.embedding_drifts = self.results['embedding_drifts']
        self.embeddings = self.results["subset_embeddings"]
        self.labels = self.results["subset_labels"]

        self.cka_similarities = None
        self.cka_similarities_denoised = {}

        self.pt_files = None
        self.flattened_weights = None

    def get_cka_similarities(self):
        if self.cka_similarities is None:
            self.cka_similarities = calculate_cka_similarities(self.embeddings)
        return self.cka_similarities

    def load(self):
        from helper.data_manager import load_training_data

        loaded_results = load_training_data(self.run_id)
        loaded_results["embedding_drifts"] = {int(k): loaded_results["embedding_drifts"][k] for k in
                                       sorted(loaded_results["embedding_drifts"].keys(), key=int)}
        return loaded_results

    def reload(self):
        self.results = self.load()
        self.embedding_drifts = self.results['embedding_drifts']
        self.embeddings = self.results["subset_embeddings"]
        self.labels = self.results["subset_labels"]

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

        return fig

    def plot_embedding_drifts(self, doubled_lines = True, y_lim=None, metric='euclidean'):
        if metric == 'euclidean':
            embedding_drifts = self.embedding_drifts.copy()
        elif metric == 'manhattan':
            embedding_drifts = calculate_embedding_drift(self.embeddings, metric='manhattan')
        else:
            raise ValueError(f"Not implemented metric {metric}")

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

        plots.plot_embedding_drift(axs, embedding_drifts, max_multiply=1.5, y_lim=y_lim)

        plt.legend()
        return fig

    def plot_embedding_drifts_manhattan(self, doubled_lines = True, y_lim=None):
        embedding_drifts = self.embedding_drifts.copy()
        embedding_drifts_manhattan = calculate_embedding_drift(self.embeddings, metric='manhattan')

        fig, axs = plt.subplots(1, 1, figsize=(10, 4))

        # Plot scaled Drifts
        n = np.array(embedding_drifts_manhattan[1]).mean() / np.array(embedding_drifts[1]).mean()
        if doubled_lines:
            axs.plot(range(1, len(embedding_drifts[1]) + 1), np.array(embedding_drifts[1]) * n, color="green",
                     label="Eucl. Drift 1", alpha=0.3)
            axs.plot(range(1, len(embedding_drifts[2]) + 1), np.array(embedding_drifts[2]) * n, color="blue",
                     label="Eucl. Drift 2", alpha=0.3)
            axs.plot(range(1, len(embedding_drifts[4]) + 1), np.array(embedding_drifts[4]) * n, color="orange",
                     label="Eucl. Drift 4", alpha=0.3)
            axs.plot(range(1, len(embedding_drifts[8]) + 1), np.array(embedding_drifts[8]) * n, color="red",
                     label="Eucl. Drift 8", alpha=0.3)
            axs.plot(range(1, len(embedding_drifts[8]) + 1), np.array(embedding_drifts[16] * n), color="purple",
                     label="Eucl. Drift 16", alpha=0.3)

        plots.plot_embedding_drift(axs, embedding_drifts_manhattan, max_multiply=1.5, y_lim=y_lim,
                                   title="Embedding Drift Manhattan vs Euclidean Distance")

        plt.legend()
        return fig

    def plot_cka_similarities(self, doubled_lines = True, flip=True, y_lim=None):
        cka_similarities = self.get_cka_similarities()
        fig, axs = plt.subplots(1, 1, figsize=(10, 4))

        if flip:
            cka_similarities = {
                k: [1 - x if not np.isnan(x) else np.nan for x in v]
                for k, v in cka_similarities.items()
            }

        # Plot 2x Drifts
        if doubled_lines:
            axs.plot(range(1, len(cka_similarities[1]) + 1), np.array(cka_similarities[1]) * 2, color="green",
                     label="2x Drift 1", alpha=0.3)
            axs.plot(range(1, len(cka_similarities[2]) + 1), np.array(cka_similarities[2]) * 2, color="blue",
                     label="2x Drift 2", alpha=0.3)
            axs.plot(range(1, len(cka_similarities[4]) + 1), np.array(cka_similarities[4]) * 2, color="orange",
                     label="2x Drift 2", alpha=0.3)
            axs.plot(range(1, len(cka_similarities[8]) + 1), np.array(cka_similarities[8]) * 2, color="red",
                     label="2x Drift 2", alpha=0.3)

        plots.plot_embedding_drift(axs, cka_similarities,
                                   title=f"CKA Similarities{' (1 - Sim)' if flip else ''}",
                                   max_multiply=1.5,
                                   y_lim=y_lim)

        plt.legend()
        return fig

    def plot_curvature_distribution(self):
        """
        Computes per-sample curvature of embedding trajectories over time
        and plots mean ± standard deviation.
        """
        embeddings = np.array(self.embeddings)  # shape: (T, N, D)
        T = embeddings.shape[0]

        curvature_matrix = []

        for t in tqdm(range(1, T - 1), desc="Computing curvature distribution"):
            delta_prev = embeddings[t] - embeddings[t - 1]
            delta_next = embeddings[t + 1] - embeddings[t]

            dot_products = np.sum(delta_prev * delta_next, axis=1)
            norms_prev = np.linalg.norm(delta_prev, axis=1)
            norms_next = np.linalg.norm(delta_next, axis=1)
            denom = norms_prev * norms_next + 1e-8

            cos_angles = np.clip(dot_products / denom, -1.0, 1.0)
            angles = np.arccos(cos_angles)  # shape: (N,)
            curvature_matrix.append(angles)

        curvature_matrix = np.array(curvature_matrix)  # shape: (T-2, N)

        mean_curv = curvature_matrix.mean(axis=1)
        std_curv = curvature_matrix.std(axis=1)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(range(1, T - 1), mean_curv, label="Mean Curvature")
        plt.fill_between(range(1, T - 1), mean_curv - std_curv, mean_curv + std_curv,
                         alpha=0.3, label="±1 Std Dev")
        plt.xlabel("Epoch")
        plt.ylabel("Trajectory Curvature (radians)")
        plt.title("Mean ± Std of Trajectory Curvature")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        return fig

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
        self.cka_similarities = calculate_cka_similarities(self.embeddings)

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

    def eigenvalues(self, num_components = 10, figsize=(10, 6)):
        from sklearn.decomposition import PCA

        eigenvalues_over_time = []

        for E_t in self.embeddings:  # E_t has shape (N, D)
            pca = PCA(n_components=num_components)
            pca.fit(E_t)
            eigenvalues_over_time.append(pca.explained_variance_)

        eigenvalues_over_time = np.array(eigenvalues_over_time)  # shape (T, num_components)

        # Plot
        fig = plt.figure(figsize=figsize)
        for i in range(num_components):
            plt.plot(eigenvalues_over_time[:, i], label=f'PC {i + 1}')
        plt.xlabel("Epoch")
        plt.ylabel("Eigenvalue (Explained Variance)")
        plt.title("Top 10 PCA Eigenvalues Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        return fig

    def plot_embedding_drift_single(self):
        """
        Plots mean ± std of embedding drift (Euclidean) with skip=1 over time.
        """
        embeddings = np.array(self.embeddings)  # shape: (T, N, D)
        T = embeddings.shape[0]

        drift_vals = []

        for t in range(1, T):
            delta = embeddings[t] - embeddings[t - 1]
            drift = np.linalg.norm(delta, axis=1)  # shape: (N,)
            drift_vals.append(drift)

        drift_vals = np.array(drift_vals)  # shape: (T-1, N)
        mean_drift = drift_vals.mean(axis=1)
        std_drift = drift_vals.std(axis=1)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(range(1, T), mean_drift, label="Mean Drift", color="tab:blue")
        plt.fill_between(range(1, T), mean_drift - std_drift, mean_drift + std_drift,
                         alpha=0.3, label="±1 Std Dev", color="tab:blue")
        plt.xlabel("Epoch")
        plt.ylabel("Drift Distance")
        plt.title("Mean ± Std of Embedding Drift (Skip=1)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        return fig

    def plot_embedding_drift_multi(self):
        """
        Plots mean ± std of embedding drift (Euclidean) for multiple skip values in one figure.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        embeddings = np.array(self.embeddings)  # shape: (T, N, D)
        T = embeddings.shape[0]
        skips = [1, 2, 4, 8, 16]

        fig = plt.figure(figsize=(10, 6))
        colors = plt.get_cmap("tab10").colors

        for idx, skip in enumerate(skips):
            if T <= skip:
                continue  # skip too large for sequence length
            drift_vals = []

            for t in range(skip, T):
                delta = embeddings[t] - embeddings[t - skip]
                drift = np.linalg.norm(delta, axis=1)  # shape: (N,)
                drift_vals.append(drift)

            drift_vals = np.array(drift_vals)  # shape: (T - skip, N)
            mean_drift = drift_vals.mean(axis=1)
            std_drift = drift_vals.std(axis=1)
            x = range(skip, T)

            plt.plot(x, mean_drift, label=f"Skip {skip}", color=colors[idx % len(colors)])
            plt.fill_between(x, mean_drift - std_drift, mean_drift + std_drift,
                             alpha=0.3, color=colors[idx % len(colors)])

        plt.xlabel("Epoch")
        plt.ylabel("Drift Distance")
        plt.title("Embedding Drift: Mean ± Std for Multiple Skips")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        return fig

    def get_pt_files(self, model_folder="trainings/{}", load_only=False):
        if self.pt_files is not None:
            return self.pt_files

        model_folder = model_folder.format(self.results["ll_flattened_weights_dir"])
        self.pt_files = get_files(model_folder, prefix="model-")
        print(f"Found {len(self.pt_files)} checkpoint files.")

        if load_only:
            return
        return self.pt_files

    def get_flattened_weights(self):
        if self.flattened_weights is not None:
            return self.flattened_weights

        self.flattened_weights = load_flattened_weights(self.get_pt_files())
        return self.flattened_weights



class Animation:
    def __init__(self, projections: list, title: str, run: Run):
        self.projections = projections
        self.title = title
        # From Run
        self.run = run
        self.run_id = run.run_id
        self.labels = run.labels
        self.embedding_drifts = run.embedding_drifts.copy()
        self.cka_similarities = run.cka_similarities.copy() if run.cka_similarities is not None else None

        self.meta = dict()
        self.meta["denoised"] = False

    def copy(self):
        return Animation(self.projections.copy(), self.title, self.run)

    def save(self, file_title=None):
        from helper.data_manager import save_animation
        self.get_cka_similarities() # Ensure to save CKA Similarities too
        save_animation(self, file_title)
        return self

    def get_cka_similarities(self):
        if self.cka_similarities is None:
            self.cka_similarities = self.run.get_cka_similarities()
        return self.cka_similarities

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
        filename = f"plots/{self.run_id}/{self.title}.gif"

        ani.save(filename, writer='pillow', dpi=150)
        plt.close(ani._fig)

        print(filename)

    def evaluate(self, verbose=True, figsize=(10, 3), y_lim=None, metric="euclidean", is_trajectory=False):
        return visualization_drift_vs_embedding_drift(
            self.projections,
            self.embedding_drifts if metric == 'euclidean' else calculate_embedding_drift(self.run.embeddings, metric=metric),
            self.get_cka_similarities(),
            verbose=verbose,
            figsize=figsize,
            y_lim=y_lim,
            axis=0 if is_trajectory else 1
        )

    def log_evaluation(self, is_trajectory=False):
        return visualization_drift_vs_embedding_drift(
            self.projections,
            self.embedding_drifts,
            self.get_cka_similarities(),
            verbose=False,
            axis=0 if is_trajectory else 1,
            logging=True
        )

    def denoise(self, blend=0.9, window_size=15, mode='window', do_projections=True, do_embedding_drift=True,
                do_cka_similarities=True):
        copy = self.copy()

        copy.meta = {
            "denoised": True,
            "blend": blend,
            "window_size": window_size,
            "mode": mode,
            "do_projections": do_projections,
            "do_embedding_drift": do_embedding_drift,
            "do_cka_similarities": do_cka_similarities
        }

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

        if do_cka_similarities:
            key = get_denoise_key(mode, blend, window_size)
            if key in self.run.cka_similarities_denoised.keys():
                copy.cka_similarities = self.run.cka_similarities_denoised[key]
            else:
                print(f"As do_cka_similarities=True, compute cka similarities for denoised projections: {key}")
                self.run.cka_similarities_denoised[key] = calculate_cka_similarities(
                    denoise_projections(
                        self.run.embeddings,
                        window_size=window_size,
                        blend=blend,
                        mode=mode
                    )
                )
            copy.cka_similarities = self.run.cka_similarities_denoised[key]

        return copy

    def scatter_movements(self, skip=1, fig_size=(5.5, 5.5), combine_all=False,):
        return plots.plot_movement_scatter(
            self,
            skip=skip,
            fig_size=fig_size,
            combine_all=combine_all
        )
    def evaluate_movements(self, color_by_label=False, sample_step=2, alpha=0.2, point_size=2, fig_size_base=(3, 3.5),
                           start_frame=0):
        return plots.plot_combined_skips(
            self,
            color_by_label=color_by_label,
            sample_step=sample_step,
            alpha=alpha,
            point_size=point_size,
            fig_size_base=fig_size_base,
            start_frame=start_frame
        )

    def eval_plot(self, *args, **kwargs):
        return plots.evaluation_plot(self, *args, **kwargs)


def load_stored_animation(run: Run, title: str):
    import os
    from helper.data_manager import load_animation

    path = os.path.join("trainings", run.run_id, f"{title}.h5")
    if not os.path.exists(path):
        return None  # no cached animation

    ani = load_animation(run, title)
    print(f"Loaded cached animation: {title}")

    # Guard against missing metadata
    if not getattr(ani, "meta", None) or not ani.meta.get("denoised", False):
        return ani

    if not ani.meta.get("do_projections", True):
        return ani

    window_size = ani.meta["window_size"]
    blend = ani.meta["blend"]
    mode = ani.meta["mode"]

    if ani.meta.get("do_embedding_drift", True):
        ani.embedding_drifts = calculate_embedding_drift(
            denoise_projections(
                run.embeddings,
                window_size=window_size,
                blend=blend,
                mode=mode
            )
        )

    if ani.meta.get("do_cka_similarities", True):
        key = get_denoise_key(mode, blend, window_size)
        if key in run.cka_similarities_denoised.keys():
            ani.cka_similarities = run.cka_similarities_denoised[key]
        else:
            print(f"As do_cka_similarities=True, compute cka similarities for denoised projections: {key}")
            run.cka_similarities_denoised[key] = calculate_cka_similarities(
                denoise_projections(
                    run.embeddings,
                    window_size=window_size,
                    blend=blend,
                    mode=mode
                )
            )
        ani.cka_similarities = run.cka_similarities_denoised[key]

    return ani

def get_denoise_key(mode, blend, window_size):
    return f"{mode}-{blend}" if mode == 'exponential' else f"{mode}-{blend}-{window_size}"

def visualization_drift_vs_embedding_drift(projections, embedding_drifts, cka_similarities, verbose=True, embeddings=False,
                                           figsize=(10, 3), on_ax=None, y_lim=None, axis=1, logging=False):
    """
    Computes the composite similarity score between visualization and embedding drift
    across all skip levels, and optionally visualizes the results.
    """
    if embeddings:
        embedding_drifts = calculate_embedding_drift(embedding_drifts)
    else:
        embedding_drifts = embedding_drifts.copy()

    if type(projections) is not list and projections.ndim == 2:  # (epochs, 2)
        projections = [p[None, :] for p in projections]

    visualization_drifts = calculate_embedding_drift(projections, axis=axis)

    assert len(visualization_drifts) == len(embedding_drifts) and len(cka_similarities) == len(visualization_drifts), \
        (f"Mismatch in drift lengths. "
         f"Vis: {len(visualization_drifts)} vs Emb: {len(embedding_drifts)} vs CKA: {len(cka_similarities)}")

    # Flip Similarities
    cka_similarities = {
        k: [1 - x if not np.isnan(x) else np.nan for x in v]
        for k, v in cka_similarities.items()
    }

    # Compute similarity
    mean_drift_sim, drift_sim_scores, log_drift = compute_drift_similarity_score(
        embedding_drifts, visualization_drifts, verbose=verbose, similarity_name="Total Mean Sim. to EMBEDDING DRIFT"
    )
    mean_cka_sim, cka_sim_scores, log_cka = compute_drift_similarity_score(
        cka_similarities, visualization_drifts, verbose=verbose, similarity_name="Total Mean Sim. to CKA SIMILARITY"
    )

    if logging:
        return log_drift + log_cka

    if on_ax is not None:
        plots.plot_embedding_drift(cka_sim_scores, visualization_drifts, title="Visualization Drift")
        return drift_sim_scores, cka_sim_scores

    if verbose:
        fig, axs = plt.subplots(1, 3, figsize=figsize)
        plots.plot_embedding_drift(axs[0], visualization_drifts, title="Visualization Drift", y_lim=y_lim)
        plots.plot_embedding_drift(axs[1], embedding_drifts, y_lim=y_lim)
        plots.plot_embedding_drift(axs[2], cka_similarities, title="CKA Similarities (flipped)",)
        plt.legend()
        plt.show()

    return mean_drift_sim, mean_cka_sim


def calculate_embedding_drift(embedding_snapshots, max_power=5, metric='euclidean', axis=1):
    """
    Calculate embedding drift based on the snapshots.
    Drift is calculated as the mean distance (Euclidean or Manhattan) between snapshots.
    Uses skip steps as powers of 2 (i.e., 1, 2, 4, 8, ...).

    Parameters:
        embedding_snapshots: list or array of shape (T, N, D)
        max_power: maximum power of 2 for skip steps
        metric: 'euclidean' (L2) or 'manhattan' (L1)
    """
    assert metric in ('euclidean', 'manhattan'), "Metric must be 'euclidean' or 'manhattan'"

    drifts = {2 ** n: [] for n in range(max_power)}

    for i in range(1, len(embedding_snapshots)):
        current_snapshot = embedding_snapshots[i]

        for n in range(max_power):
            skip = 2 ** n
            if i - skip >= 0:
                previous_snapshot = embedding_snapshots[i - skip]
                if metric == 'euclidean':
                    drift = np.linalg.norm(current_snapshot - previous_snapshot, axis=axis).mean()
                elif metric == 'manhattan':
                    drift = np.abs(current_snapshot - previous_snapshot).sum(axis=axis).mean()
                drifts[skip].append(drift)
            else:
                drifts[skip].append(np.nan)

    return drifts

def calculate_cka_similarities(embedding_snapshots, max_power=5):
    """
    Calculate CKA similarity between embedding snapshots over time.
    Similarity is computed between time t and t - skip, using skip steps 2^n.
    Returns a dictionary: {skip: [cka_1, cka_2, ...]} aligned with time steps.
    """
    similarities = {2 ** n: [] for n in range(max_power)}
    T = len(embedding_snapshots)

    for i in tqdm(range(1, T), desc="Calculating CKA similarities"):
        current = embedding_snapshots[i]

        for n in range(max_power):
            skip = 2 ** n
            if i - skip >= 0:
                previous = embedding_snapshots[i - skip]
                cka = compute_cka(current, previous)
                similarities[skip].append(cka)
            else:
                similarities[skip].append(np.nan)

    return similarities

def compute_drift_similarity_score(embedding_drifts,
                                   visualization_drifts,
                                   eps=1e-6,
                                   verbose=True,
                                   lambda_penalty=3.0,
                                   similarity_name="Mean Composite Similarity"):
    """
    Computes composite similarity: log-Pearson correlation × MSE-based ratio penalty.

    Args:
        embedding_drifts (dict): Drift values for high-dimensional embeddings.
        visualization_drifts (dict): Drift values for low-dimensional projections.
        eps (float): Small constant to avoid division by zero or log(0).
        verbose (bool): Whether to print per-skip values.
        lambda_penalty (float): Strength of penalty in ratio mismatch.
        similarity_name: (string)

    Returns:
        float: mean similarity across all skips
        dict: similarity per skip
    """
    similarity_scores = {}
    log = ""

    for k in embedding_drifts.keys():
        emb = np.asarray(embedding_drifts[k])
        vis = np.asarray(visualization_drifts[k])

        mask = ~np.isnan(emb) & ~np.isnan(vis)
        if np.sum(mask) < 3:
            similarity_scores[k] = np.nan
            continue

        emb = emb[mask]
        vis = vis[mask]

        # Log-Pearson correlation
        log_emb = np.log(emb + eps)
        log_vis = np.log(vis + eps)
        log_corr, _ = pearsonr(log_emb, log_vis)

        # MSE-based ratio penalty (soft relative error)
        emb_median = np.median(emb) + eps
        vis_median = np.median(vis) + eps
        emb_ratio = emb / emb_median
        vis_ratio = vis / vis_median

        rel_error = (emb_ratio - vis_ratio) / (emb_ratio + eps)
        mse = np.mean(rel_error ** 2)
        ratio_penalty = np.exp(-lambda_penalty * mse)

        similarity = log_corr * ratio_penalty
        similarity_scores[k] = similarity

        log += f"Skip {k} — log-corr: {log_corr:.3f}, ratio pen: {ratio_penalty:.3f}, total: {similarity:.3f}\n"

    mean_similarity = np.nanmean(list(similarity_scores.values()))

    log += f"{similarity_name}: {mean_similarity:.4f}\n"
    if verbose:
        print(log)
    return mean_similarity, similarity_scores, log


def show_projections_and_drift(projections_list, titles, labels, embedding_drifts, cka_similarities, embeddings=False, interpolate=False,
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
    drift_correlations = []
    cka_correlations = []
    for i, projection in enumerate(projections_list):
        drift, cka = visualization_drift_vs_embedding_drift(projection,
                                                           embedding_drifts,
                                                           cka_similarities,
                                                           verbose=False,
                                                           embeddings=embeddings,
                                                           figsize=figsize_embedding_drift,
                                                           on_ax=axes_right[i])
        drift_correlations.append(drift)
        cka_correlations.append(cka)

    plt.show()
    drift_mean = [np.mean(corrs) for corrs in drift_correlations]
    cka_mean = [np.mean(corrs) for corrs in cka_correlations]
    for i in range(nrows):
        print(f"{titles[i]}:\n"
              f"Drift: {drift_mean[i]} = {drift_correlations[i]}"
              f"CKA: {cka_mean[i]} = {cka_correlations[i]}")


def generate_pca_animation(
        run: Run,
        fit_basis='all',
        max_frames=None,
        window_size=10,
        out_dim=2,  # 2D vs 3D
        fit_basis_n=5,
):
    from sklearn.decomposition import PCA

    projections = []
    embeddings_list = run.embeddings.copy()
    max_frames = max_frames or len(embeddings_list)

    if fit_basis == 'window':
        title = f'PCA window ({window_size}){" 3D" if out_dim == 3 else ""}'

        # Try to load animation
        ani = load_stored_animation(run, title)
        if ani is not None:
            return ani

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

    else:
        # Determine basis data
        if isinstance(fit_basis, int):
            basis_data = embeddings_list[fit_basis]
        elif fit_basis == 'first':
            basis_data = embeddings_list[0]
        elif fit_basis == 'last':
            basis_data = embeddings_list[max_frames - 1]
        elif fit_basis == 'all':
            basis_data = np.concatenate(embeddings_list, axis=0)
        elif fit_basis == 'all_n':
            basis_data = np.concatenate(embeddings_list[::fit_basis_n], axis=0)
        else:
            raise ValueError(f"Invalid pca_fit_basis: {fit_basis}")

        title = f'PCA on {fit_basis} {" 3D" if out_dim == 3 else ""}'

        # Try to load animation
        ani = load_stored_animation(run, title)
        if ani is not None:
            return ani

        reducer = PCA(n_components=out_dim)
        reducer.fit(basis_data)
        for i in tqdm(range(max_frames), desc="PCA frames"):
            projections.append(reducer.transform(embeddings_list[i]))

    return Animation(projections=projections, title=title, run=run).save()


def generate_tsne_animation(
        run: Run,
        max_frames=None,
        reverse_computation=False,
        random_state=42,
        tsne_init='pca',  # 'pca' or 'random'
        tsne_perplexity=30.0,  # often between 5–50
        tsne_update=1,
        metric='euclidean',  # for umap and tsne
        out_dim=2,  # 2D vs 3D
):
    from sklearn.manifold import TSNE

    projections = []
    embeddings_list = run.embeddings.copy()
    max_frames = max_frames or len(embeddings_list)

    title = (f't-SNE (init={tsne_init}, '
            f'perplexity={tsne_perplexity}, '
            f'{metric}, '
            f'{"reverse computation, " if reverse_computation else ""}'
            f'{"" if tsne_update == 1 else f"blending={tsne_update},"}'
             f'{" 3D" if out_dim == 3 else ""})')

    # Try to load animation
    ani = load_stored_animation(run, title)
    if ani is not None:
        return ani

    print("Initializing t-SNE...")
    tsne = TSNE(n_components=out_dim, init=tsne_init, perplexity=tsne_perplexity, random_state=random_state,
                metric=metric)
    if reverse_computation:
        tsne.fit(embeddings_list[max_frames - 1]) # Last Visualized frame
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
        tsne.fit(embeddings_list[0]) # Fit on first
        projections.append(tsne.fit_transform(embeddings_list[0]))
        for i in tqdm(range(1, max_frames), desc="t-SNE frames"):
            tsne = TSNE(n_components=out_dim, init=projections[-1], perplexity=tsne_perplexity,
                        random_state=random_state, metric=metric)
            new = tsne.fit_transform(embeddings_list[i]) * tsne_update + projections[-1] * (1 - tsne_update)
            projections.append(new)


    return Animation(projections=projections, title=title, run=run).save()



def generate_umap_animation(
        run: Run,
        fit_basis='all_n',
        max_frames=None,
        reverse_computation=False,
        random_state=None,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',  # for umap and tsne
        out_dim=2,  # 2D vs 3D
        fit_basis_n=5,
):
    import umap

    projections = []
    embeddings_list = run.embeddings.copy()
    max_frames = max_frames or len(embeddings_list)

    # Determine basis data
    if fit_basis == 'first':
        basis_data = embeddings_list[0]
    elif fit_basis == 'last':
        basis_data = embeddings_list[max_frames - 1]
    elif fit_basis == 'all':
        basis_data = np.concatenate(embeddings_list, axis=0)
    elif fit_basis == 'all_n':
        basis_data = np.concatenate(embeddings_list[::fit_basis_n], axis=0)
    else:
        raise ValueError(f"Invalid pca_fit_basis: {fit_basis}")

    title = (f'UMAP (n={n_neighbors}, dist={min_dist}, {metric}, '
            f'{"reverse computation, " if reverse_computation else ""}'
             f'{" 3D" if out_dim == 3 else ""})')

    # Try to load animation
    ani = load_stored_animation(run, title)
    if ani is not None:
        return ani

    reducer = umap.UMAP(n_components=out_dim,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        metric=metric,
                        random_state=random_state)
    if reverse_computation:
        for i in tqdm(range(max_frames - 1, -1, -1), desc="UMAP frames"):
            reducer.fit(embeddings_list[i])
            projection = reducer.transform(embeddings_list[i])
            projections.insert(0, projection)
    else:
        print(f"Fit UMAP on basis data: {fit_basis}")
        reducer.fit(basis_data)
        for i in tqdm(range(max_frames), desc="UMAP frames"):
            projection = reducer.transform(embeddings_list[i])
            projections.append(projection)

    return Animation(projections=projections, title=title, run=run).save()

def generate_phate_animation(
        run: Run,
        max_frames=None,
        out_dim=2,
        knn=5,
        decay=40,
        t=20,
        n_jobs=-1,
        random_state=42,
        window=0  # number of frames before to include for fitting
):
    import phate
    import numpy as np
    from tqdm import tqdm
    from scipy.linalg import orthogonal_procrustes

    embeddings_list = run.embeddings.copy()
    max_frames = max_frames or len(embeddings_list)
    projections = []

    title = f'PHATE (knn={knn}, decay={decay}, t={t}, window={window})'

    prev_projection = None

    for i in tqdm(range(max_frames), desc="PHATE frames"):
        # Determine window range
        start = max(0, i - window)
        end = min(max_frames, i + 1)
        fit_data = np.concatenate(embeddings_list[start:end], axis=0)

        # Fit PHATE on the windowed data
        phate_op = phate.PHATE(
            n_components=out_dim,
            knn=knn,
            decay=decay,
            t=t,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=False
        )
        fit_projection = phate_op.fit_transform(fit_data)

        # Extract just the projection for the current frame
        current_len = len(embeddings_list[i])
        offset = sum(len(embeddings_list[j]) for j in range(start, i))
        projection = fit_projection[offset:offset + current_len]

        # Flip and align using orthogonal Procrustes
        if prev_projection is not None:
            R, _ = orthogonal_procrustes(projection, prev_projection)
            projection = projection @ R

        projections.append(projection)
        prev_projection = projection.copy()

    return Animation(projections=projections, title=title, run=run)

def generate_mphate_animation(run: Run, title="M-PHATE", t='auto', gamma=0, interslice_knn=25, *args, **kwargs):
    file_title = f"M-PHATE gm={gamma} knn={interslice_knn} t={t}"

    ani = load_stored_animation(run, file_title)
    if ani is not None:
        ani.title = title
        return ani

    mphate_emb = compute_mphate_embeddings(run, gamma=gamma, interslice_knn=interslice_knn, t=t, *args, **kwargs)
    projections = [mphate_emb[i] for i in range(mphate_emb.shape[0])]

    return Animation(projections=projections, title=title, run=run).save(file_title=file_title)


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
        annotate_confusion_matrix=False,
        cols=None,
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
            annotate_conf=annotate_confusion_matrix,
            cols=cols
        )
        return

    embedding_drifts = [ani.embedding_drifts for ani in animations]
    cka_similarities = [ani.get_cka_similarities() for ani in animations]
    show_projections_and_drift(
        projections_list = projections_list,
        titles = titles,
        labels=animations[0].labels,
        embedding_drifts= embedding_drifts,
        cka_similarities=cka_similarities,
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

def compute_mphate_embeddings(run, n_components=2, random_state=42, verbose=True, intraslice_knn=2, interslice_knn=25,
                              decay=5, t='auto', gamma=0):
    """
    Computes M-PHATE embeddings from a run object.

    Parameters:
        run: Run object with .embeddings (list of arrays with shape [samples, features])
        n_components: Target dimensionality (default: 2)
        random_state: Seed for reproducibility
        verbose: If True, prints progress info

    Returns:
        mphate_emb: np.ndarray of shape (epochs, samples, n_components)
    """
    import m_phate

    emb_list = run.embeddings
    n_epochs = len(emb_list)
    n_samples = emb_list[0].shape[0]

    if verbose:
        print(f"n_epochs: {n_epochs}")
        print(f"n_samples: {n_samples}")

    emb_tensor = np.stack(emb_list)  # shape: (epochs, samples, features)

    mphate_op = m_phate.M_PHATE(
        n_components=n_components,
        intraslice_knn=intraslice_knn,  # Local structure sensitivity within each epoch (default 2  or 2-5)
        interslice_knn=interslice_knn,  # Controls smoothness across time steps (default 25 or 15-30)
        decay=decay,                    # Controls kernel sharpness (edge strength in affinities) (default 5 or 3-10)
        t=t,                            # Diffusion scale: controls local vs. global structure (default 'auto' or 10–30)
        gamma=gamma,                    # Influences distance potential (PHATE vs. sqrt behavior) (default 0 or 0-0.2)
        random_state=random_state,
        verbose=verbose
    )
    mphate_flat = mphate_op.fit_transform(emb_tensor)  # shape: (epochs * samples, n_components)

    mphate_emb = mphate_flat.reshape(n_epochs, n_samples, n_components)
    return mphate_emb


def mphate_to_animation(mphate_emb, run, title="M-PHATE projection"):
    """
    Converts M-PHATE embedding array into an Animation object.

    Parameters:
        mphate_emb: np.ndarray of shape (T, N, D)
        run: the original Run object
        title: optional title for the animation

    Returns:
        Animation object
    """
    projections = [mphate_emb[i] for i in range(mphate_emb.shape[0])]
    return Animation(projections=projections, title=title, run=run)


def mphate_on_runs(runs, titles=None):
    """
    Generate M-PHATE projections and Animation objects for a set of training runs.
    Optionally use custom titles per run.
    """
    import m_phate
    all_run_flattened = []

    for run in runs:
        emb_list = run.embeddings
        flattened_epochs = [emb.reshape(-1) for emb in emb_list]
        flattened_tensor = np.stack(flattened_epochs)
        all_run_flattened.append(flattened_tensor)

    all_run_flattened = np.stack(all_run_flattened)
    combined_emb = np.transpose(all_run_flattened, (1, 0, 2))

    mphate_op = m_phate.M_PHATE()
    mphate_emb = mphate_op.fit_transform(combined_emb)

    n_epochs, n_runs = combined_emb.shape[:2]
    mphate_emb = mphate_emb.reshape(n_epochs, n_runs, 2)
    mphate_trajectories = np.transpose(mphate_emb, (1, 0, 2))

    animations = []
    for idx, run in enumerate(runs):
        title = titles[idx] if titles is not None else run.results["train_config"]
        anim = Animation(
            projections=mphate_trajectories[idx],
            title=title,
            run=run
        )
        animations.append(anim)

    return animations


def mphate_on_predictions(runs, titles=None):
    """
    Apply M-PHATE to the prediction distributions (val_distributions) across runs.
    Returns a list of Animation objects, one per run, with projections over epochs.
    """
    import m_phate

    all_run_flattened = []

    for run in runs:
        preds_per_epoch = run.results["val_distributions"]  # list of (samples, classes)
        flattened_epochs = [pred.reshape(-1) for pred in preds_per_epoch]  # shape: (samples * classes,)
        flattened_tensor = np.stack(flattened_epochs)  # shape: (epochs, flat_dim)
        all_run_flattened.append(flattened_tensor)

    all_run_flattened = np.stack(all_run_flattened)  # shape: (n_runs, epochs, flat_dim)
    combined_pred = np.transpose(all_run_flattened, (1, 0, 2))  # (epochs, runs, features)

    # Run M-PHATE
    mphate_op = m_phate.M_PHATE(knn_dist="cosine", mds_dist="cosine")
    mphate_emb = mphate_op.fit_transform(combined_pred)  # shape: (epochs * runs, 2)

    n_epochs, n_runs = combined_pred.shape[:2]
    mphate_emb = mphate_emb.reshape(n_epochs, n_runs, 2)
    mphate_trajectories = np.transpose(mphate_emb, (1, 0, 2))  # shape: (runs, epochs, 2)

    # Wrap into Animation objects
    animations = []
    for idx, run in enumerate(runs):
        title = titles[idx] if titles is not None else run.results["train_config"]
        anim = Animation(
            projections=mphate_trajectories[idx],
            title=title,
            run=run
        )
        animations.append(anim)

    return animations

def compute_prediction_similarities(runs, similarity="cosine"):
    """
    Compute pairwise similarities of prediction distributions across multiple runs over time.

    Args:
        runs: List of Run objects, each with results["val_distributions"] as a list of (samples, num_classes)
        similarity: 'cosine' (default) or 'l2'

    Returns:
        similarities: List of similarity matrices, each of shape (n_runs, n_runs) for each epoch
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.spatial.distance import cdist

    n_runs = len(runs)
    n_epochs = len(runs[0].results["val_distributions"])
    similarities = []

    for epoch in range(n_epochs):
        epoch_vectors = []
        for run in runs:
            dist = run.results["val_distributions"][epoch]  # shape: (samples, classes)
            flat = dist.reshape(-1)
            epoch_vectors.append(flat)

        epoch_vectors = np.stack(epoch_vectors)  # shape: (n_runs, flattened)

        if similarity == "cosine":
            sim = cosine_similarity(epoch_vectors)
        elif similarity == "l2":
            sim = -cdist(epoch_vectors, epoch_vectors, metric="euclidean")
        else:
            raise ValueError(f"Unsupported similarity metric: {similarity}")

        similarities.append(sim)

    return similarities  # List of (n_runs, n_runs)


def compute_cka(X, Y):
    """
    Compute the linear Centered Kernel Alignment (CKA) similarity between two embedding matrices X and Y.

    Parameters:
        X, Y: numpy arrays of shape (N, D)
              Rows are samples (e.g., data points), columns are features (e.g., neurons).
              Both matrices must have the same number of samples (rows).

    Returns:
        cka_score: Scalar in [0, 1] measuring similarity between the representations.
                   A value close to 1 indicates strong structural alignment.
    """
    from sklearn.metrics.pairwise import linear_kernel

    def center_kernel(K):
        """Double-center the Gram matrix K using the centering matrix H = I - 1/n."""
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    # Compute linear Gram (kernel) matrices
    Kx = linear_kernel(X)  # Kx[i,j] = <x_i, x_j>
    Ky = linear_kernel(Y)  # Ky[i,j] = <y_i, y_j>

    # Center the kernel matrices
    Kx_centered = center_kernel(Kx)
    Ky_centered = center_kernel(Ky)

    # Compute the Hilbert-Schmidt Independence Criterion (HSIC)
    hsic = np.sum(Kx_centered * Ky_centered)

    # Normalize to obtain CKA
    norm_Kx = np.linalg.norm(Kx_centered, 'fro')
    norm_Ky = np.linalg.norm(Ky_centered, 'fro')
    cka_score = hsic / (norm_Kx * norm_Ky + 1e-8)

    return cka_score

from helper.visualization import Animation

def mphate_on_model_weights(runs: list[Run], titles=None):
    """
    Apply M-PHATE to flattened model weights from multiple runs.

    Args:
        runs: List of runs
        titles: optional list of labels per run

    Returns:
        animations: list of Animation objects, one per run
    """
    import m_phate

    all_run_flattened = np.stack([run.get_flattened_weights() for run in runs])  # shape: (n_runs, n_epochs, features)
    print(f'shape: (n_runs, n_epochs, features): {all_run_flattened.shape}')

    combined_weights = np.transpose(all_run_flattened, (1, 0, 2))  # shape: (epochs, runs, features)
    print(f'shape: (epochs, runs, features): {combined_weights.shape}')

    mphate_op = m_phate.M_PHATE(knn_dist="cosine", mds_dist="cosine")
    mphate_emb = mphate_op.fit_transform(combined_weights)  # shape: (epochs * runs, 2)

    n_epochs, n_runs = combined_weights.shape[:2]
    mphate_emb = mphate_emb.reshape(n_epochs, n_runs, 2)
    mphate_trajectories = np.transpose(mphate_emb, (1, 0, 2))  # (runs, epochs, 2)

    animations = []
    for idx, run in enumerate(runs):
        title = titles[idx] if titles else f"Run {idx}"
        anim = Animation(
            projections=mphate_trajectories[idx],
            title=title,
            run=run
        )
        animations.append(anim)

    return animations


def get_files(file_path, num_models=None, prefix="", from_last=False, every_nth=1):
    """
        Copied from NeuroVisualizer.aux.utils
    """
    import re
    import os

    def extract_number(s, prefix=prefix):
        pattern = re.compile(r'{}(\d+).pt'.format(prefix))
        match = pattern.search(s)
        if match:
            return int(match.group(1))
        else:
            return float('inf')

    def get_all_files(d):
        f_ = []
        for dirpath, dirnames, filenames in os.walk(d):
            f_temp = []
            for filename in filenames:
                f_temp.append(os.path.join(dirpath, filename))

            f_temp = [file for file in f_temp if os.path.splitext(file)[-1] == ".pt"]
            f_temp = sorted(f_temp, key=extract_number)

            len_f_temp_original = len(f_temp)
            print(dirpath, 'has', len_f_temp_original, 'files')

            if every_nth > 1 and len_f_temp_original > 0:
                f_temp_last = f_temp[-1]
                f_temp = f_temp[::every_nth]

                if len_f_temp_original % every_nth != 1:
                    f_temp = f_temp + [f_temp_last]

            f_ = f_ + f_temp
        return f_

    directory = os.path.join(file_path)
    files = get_all_files(directory)
    pt_files = [file for file in files if file.endswith(".pt")]
    if num_models is not None:
        pt_files = pt_files[:num_models] if not from_last else pt_files[-num_models:]

    return pt_files


def load_flattened_weights(pt_file_paths, device="cpu"):
    """
    Load already-flattened model weights (using weights_only=True) from .pt files.
    """
    import torch
    flattened = []
    for path in tqdm(pt_file_paths, desc="Loading model checkpoints"):
        try:
            tensor = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            raise ValueError(f"torch.load(..., weights_only=True) is not supported for {path}.")

        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Expected a flattened tensor in {path}, got {type(tensor)}")

        flattened.append(tensor.detach().cpu().numpy())

    return np.stack(flattened)  # shape: (n_checkpoints, total_weights)

def linear_mode_connectivity_path(run1, run2, num_points=50):
    """
    Compute a straight‐line (LMC) path in weight‐space between two trained models.

    Args:
        run1, run2: Run objects, each must implement
            get_flattened_weights() -> np.ndarray of shape (n_epochs, n_features)
        num_points: number of interpolation points (including the endpoints)

    Returns:
        path: list of np.ndarray, each of shape (n_features,)
              path[0] == final weights of run1,
              path[-1] == final weights of run2
    """
    # grab the final epoch's flattened weights from each run
    w1 = run1.get_flattened_weights()[-1]  # shape: (features,)
    w2 = run2.get_flattened_weights()[-1]  # shape: (features,)

    # alphas from 0 to 1
    alphas = np.linspace(0.0, 1.0, num_points)

    # build the path: (1−α_i)*w1 + α_i*w2
    path = [(1 - a) * w1 + a * w2 for a in alphas]
    return path

def _get_epoch_predictions(run, t, *, ensure_prob=True, eps=1e-12):
    """
    Return predictions for epoch t as (N, C) float array.
    If ensure_prob=True, re-normalizes along classes to be safe.
    """
    P = np.asarray(run.results["val_distributions"][t], dtype=float)
    if ensure_prob:
        P = np.maximum(P, eps)
        P = P / P.sum(axis=1, keepdims=True)
    return P

def _pred_similarity(X, Y, *, metric="cosine", eps=1e-12):
    """
    Similarity between two prediction sets X, Y of shapes (N, C) & (M, C).
    We truncate to the smallest N and compare aligned samples.
    Metrics:
      - "cosine": cosine similarity on flattened vectors in [-1, 1]
      - "js":     Jensen-Shannon similarity in [0, 1], defined as 1 - JS / log(2)
    """
    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]

    if metric == "cosine":
        x = X.reshape(1, -1)
        y = Y.reshape(1, -1)
        num = float(np.dot(x, y.T))
        den = float(np.linalg.norm(x) * np.linalg.norm(y) + eps)
        return num / den

    if metric == "js":
        # Per-sample JS divergence averaged over samples
        # JS(P, Q) = 0.5 * [KL(P||M) + KL(Q||M)], with M = 0.5*(P+Q)
        P = np.clip(X, eps, 1.0)
        Q = np.clip(Y, eps, 1.0)
        M = 0.5 * (P + Q)
        kl_pm = (P * (np.log(P) - np.log(M))).sum(axis=1)
        kl_qm = (Q * (np.log(Q) - np.log(M))).sum(axis=1)
        js = 0.5 * (kl_pm + kl_qm)            # in nats
        js_bits = js / np.log(2.0)            # normalize by log(2) for [0, 1] scale
        sim = 1.0 - float(np.mean(js_bits))   # similarity: 1 (identical) → 0 (orthogonal)
        return sim

    raise ValueError("metric must be 'cosine' or 'js'")

def compute_prediction_cross_epoch_similarity(
    run_x,
    run_y,
    *,
    metric="cosine",          # "cosine" or "js"
    skip=1,
    start_epoch=0,
    desc_prefix=""
):
    """
    Compute a coarse cross-epoch similarity grid between two runs' predictions.
    Returns:
      S  : (len(ix), len(iy)) similarity values
      ix : sampled epoch indices for X (rows)
      iy : sampled epoch indices for Y (cols)
    """
    if skip < 1:
        raise ValueError("skip must be >= 1")

    Tx = len(run_x.results["val_distributions"])
    Ty = len(run_y.results["val_distributions"])

    ix = list(range(start_epoch, Tx, skip))
    iy = list(range(start_epoch, Ty, skip))

    preds_x = [_get_epoch_predictions(run_x, t) for t in ix]
    preds_y = [_get_epoch_predictions(run_y, s) for s in iy]

    S = np.zeros((len(ix), len(iy)), dtype=float)
    desc = f"{desc_prefix}Cross-epoch prediction similarity [{metric}] (skip={skip}, start={start_epoch})"
    for a, Px in tqdm(list(enumerate(preds_x)), desc=desc):
        for b, Py in enumerate(preds_y):
            S[a, b] = _pred_similarity(Px, Py, metric=metric)

    return S, ix, iy

def compute_epochwise_embedding_cka(runs, skip=1):
    """
    For each epoch t, compute an (n_runs x n_runs) matrix where entry (i,j)
    is the CKA similarity between runs[i].embeddings[t] and runs[j].embeddings[t].
    Returns: list of matrices, one per epoch (like your prediction heatmaps).
    """
    n_runs = len(runs)
    n_epochs = min(len(r.embeddings) for r in runs)

    similarities = []
    for t in tqdm(range(n_epochs)[::skip], desc="Computing epoch-wise embedding CKA"):
        # Gather embeddings for this epoch
        feats = [np.asarray(r.embeddings[t]) for r in runs]
        n_samples = min(f.shape[0] for f in feats)
        feats = [f[:n_samples] for f in feats]

        # Build symmetric CKA matrix
        M = np.eye(n_runs, dtype=float)
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                cij = compute_cka(feats[i], feats[j])
                M[i, j] = M[j, i] = float(cij)
        similarities.append(M)

    return similarities

def _get_epoch_matrix(run, t, mode):
    if mode == "embeddings":
        # shape: (samples, features)
        return np.asarray(run.embeddings[t])
    elif mode == "predictions":
        # list entry is (samples, classes); flatten to 1D (samples*classes)
        dist = run.results["val_distributions"][t]  # shape: (N, C)  :contentReference[oaicite:3]{index=3}
        return dist.reshape(-1, 1)  # (NC, 1) so cosine works (treat as a vector)
    else:
        raise ValueError("mode must be 'embeddings' or 'predictions'")

def _sim(X, Y, similarity, mode):
    from sklearn.metrics.pairwise import cosine_similarity
    if similarity == "cka":
        # Expect (N, D). Align by samples if needed.
        n = min(len(X), len(Y))
        return float(compute_cka(X[:n], Y[:n]))  # 0..1  :contentReference[oaicite:4]{index=4}
    elif similarity == "cosine":
        # Expect flattened vectors for predictions
        x = X.reshape(1, -1)
        y = Y.reshape(1, -1)
        return float(cosine_similarity(x, y)[0, 0])
    else:
        raise ValueError("similarity must be 'cka' or 'cosine'")

def compute_cross_epoch_similarity(
    run_x,
    run_y,
    *,
    mode="embeddings",        # "embeddings" or "predictions"
    similarity="cka",         # "cka" (embeddings) or "cosine" (predictions)
    skip=1,
    start_epoch=0,
    desc_prefix=""
):
    """
    Compute a coarse cross-epoch similarity grid between two runs.

    Returns
    -------
    S : np.ndarray, shape (len(ix), len(iy))
        Similarity values for sampled epochs of X (rows) vs Y (cols).
    ix, iy : list[int]
        Actual epoch indices used for X and Y respectively.
    """
    if skip < 1:
        raise ValueError("skip must be >= 1")

    Tx = len(run_x.embeddings) if mode == "embeddings" else len(run_x.results["val_distributions"])
    Ty = len(run_y.embeddings) if mode == "embeddings" else len(run_y.results["val_distributions"])

    # Sampled epoch indices after start_epoch
    ix = list(range(start_epoch, Tx, skip))
    iy = list(range(start_epoch, Ty, skip))

    # Pre-cache sampled epoch features/vectors
    feats_x = [_get_epoch_matrix(run_x, t, mode) for t in ix]
    feats_y = [_get_epoch_matrix(run_y, s, mode) for s in iy]

    # Compute coarse similarity grid
    S = np.zeros((len(ix), len(iy)), dtype=float)
    desc = f"{desc_prefix}Cross-epoch {similarity.upper()} ({mode}) [skip={skip}, start={start_epoch}]"
    for a, Xi in tqdm(list(enumerate(feats_x)), desc=desc):
        for b, Yj in enumerate(feats_y):
            S[a, b] = _sim(Xi, Yj, similarity=similarity, mode=mode)

    return S, ix, iy