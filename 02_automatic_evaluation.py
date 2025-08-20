import os
import argparse
import pandas as pd

from helper.visualization import Run, generate_pca_animation, Animation, generate_tsne_animation, \
    generate_umap_animation, generate_mphate_animation
import matplotlib.pyplot as plt


class CSVGridLogger:
    def __init__(self, path):
        """
        Initialize the logger. Loads existing CSV or starts fresh.
        """
        self.path = path
        if os.path.exists(path):
            self.df = pd.read_csv(path, index_col=0)
        else:
            self.df = pd.DataFrame()

    def set(self, run_id, column, value):
        """
        Set a value at the specified (run_id, column) position.
        Adds row/column if missing.
        """
        if column not in self.df.columns:
            self.df[column] = pd.NA  # Add new column with missing values
        if run_id not in self.df.index:
            self.df.loc[run_id] = pd.Series(dtype=object)  # Create empty row
        self.df.loc[run_id, column] = value

    def set_ani(self, run: Run, animation: Animation):
        self.set(run.run_id, animation.title, animation.log_evaluation())

    def save(self):
        """
        Write current state to CSV.
        """
        self.df.to_csv(self.path)



# ========== CONFIGURATION ========== #
def parse_args():
    parser = argparse.ArgumentParser(description="Run visualization pipeline")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., 'mnist'")
    parser.add_argument("--run_id", type=str, required=True, help="Run identifier")
    #parser.add_argument("--output_dir", type=str, default="results", help="Where to save outputs")
    #parser.add_argument("--vis_method", type=str, default="umap", choices=["pca", "tsne", "umap", "phate", "mphate"], help="Dimensionality reduction method")
    #parser.add_argument("--skip", type=int, default=1, help="Epoch skip for drift metrics")
    return parser.parse_args()


# ========== MAIN FUNCTION ========== #
def init_run():
    args = parse_args()
    print(f"Running visualization for: {args.run_id} on dataset {args.dataset}")

    run = Run(args.run_id, args.dataset)

    os.makedirs(f"plots/{run.run_id}/", exist_ok=True)

    return run, args

def general_plots(run: Run, args):
    fig = run.plot_training_records()
    fig.savefig(f"plots/{run.run_id}/training.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)

    fig = run.plot_embedding_drifts()
    fig.savefig(f"plots/{run.run_id}/embedding_drifts.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)

    fig = run.plot_embedding_drift_multi()
    fig.savefig(f"plots/{run.run_id}/embedding_drifts_std.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)


    fig = run.plot_embedding_drifts_manhattan()
    fig.savefig(f"plots/{run.run_id}/embedding_drifts_man.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)

    fig = run.plot_cka_similarities(y_lim=0.3)
    fig.savefig(f"plots/{run.run_id}/cka_similarities.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)

    fig = run.eigenvalues(figsize=(8, 5))
    fig.savefig(f"plots/{run.run_id}/eigenvalue_development.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)

    fig = run.plot_curvature_distribution()
    fig.savefig(f"plots/{run.run_id}/curvature_distribution.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)

def evaluate_pca(run: Run, args):
    logger = CSVGridLogger("plots/results_pca.csv")

    ani_pca_first = generate_pca_animation(run, fit_basis='first')
    ani_pca_last = generate_pca_animation(run, fit_basis='last')
    ani_pca_all = generate_pca_animation(run, fit_basis='all')
    ani_pca_window = generate_pca_animation(run, fit_basis='window', window_size=16)

    ani_pca_first.save_as_gif()
    ani_pca_last.save_as_gif()
    ani_pca_all.save_as_gif()
    ani_pca_window.save_as_gif()
    save_animation_eval(ani_pca_first)
    save_animation_eval(ani_pca_last)
    save_animation_eval(ani_pca_all)
    save_animation_eval(ani_pca_window)

    logger.set_ani(run, ani_pca_first)
    logger.set_ani(run, ani_pca_last)
    logger.set_ani(run, ani_pca_all)
    logger.set_ani(run, ani_pca_window)

    ani_pca_first_denoised = ani_pca_first.denoise()
    ani_pca_last_denoised = ani_pca_last.denoise()
    ani_pca_all_denoised = ani_pca_all.denoise()
    ani_pca_window_denoised = ani_pca_window.denoise()

    ani_pca_first_denoised.save_as_gif()
    ani_pca_last_denoised.save_as_gif()
    ani_pca_all_denoised.save_as_gif()
    ani_pca_window_denoised.save_as_gif()
    save_animation_eval(ani_pca_first_denoised)
    save_animation_eval(ani_pca_last_denoised)
    save_animation_eval(ani_pca_all_denoised)
    save_animation_eval(ani_pca_window_denoised)

    logger.set_ani(run, ani_pca_first_denoised)
    logger.set_ani(run, ani_pca_last_denoised)
    logger.set_ani(run, ani_pca_all_denoised)
    logger.set_ani(run, ani_pca_window_denoised)

    # 3D

    pca_all_3d = generate_pca_animation(run, fit_basis='all', out_dim=3)
    pca_win_3d = generate_pca_animation(run, fit_basis='window', window_size=16, out_dim=3)

    logger.set_ani(run, pca_all_3d)
    logger.set_ani(run, pca_win_3d)

    logger.set_ani(run, pca_all_3d.denoise())
    logger.set_ani(run, pca_win_3d.denoise())

    logger.save()


def evaluate_tsne(run: Run, args):
    logger = CSVGridLogger("plots/results_tsne.csv")

    tsne_animation = generate_tsne_animation(run)
    tsne_denoised = tsne_animation.denoise(blend=0.8, mode='exponential')

    tsne_blended_03 = generate_tsne_animation(run, tsne_update=0.3)
    tsne_blended_02 = generate_tsne_animation(run, tsne_update=0.2)
    tsne_cosine_03 = generate_tsne_animation(run, tsne_update=0.3, metric='cosine')
    tsne_cosine_02 = generate_tsne_animation(run, tsne_update=0.2, metric='cosine')
    tsne_p_5 = generate_tsne_animation(run, tsne_perplexity=5, metric='cosine')
    tsne_p_10 = generate_tsne_animation(run, tsne_perplexity=10, metric='cosine')
    tsne_p_30 = generate_tsne_animation(run, tsne_perplexity=30, metric='cosine')
    tsne_p_50 = generate_tsne_animation(run, tsne_perplexity=50, metric='cosine')

    tsne_animation.save_as_gif()
    tsne_denoised.save_as_gif()
    tsne_blended_03.save_as_gif()
    tsne_blended_02.save_as_gif()
    tsne_cosine_03.save_as_gif()
    tsne_cosine_02.save_as_gif()
    tsne_p_5.save_as_gif()
    tsne_p_10.save_as_gif()
    tsne_p_30.save_as_gif()
    tsne_p_50.save_as_gif()
    save_animation_eval(tsne_animation)
    save_animation_eval(tsne_denoised)
    save_animation_eval(tsne_blended_03)
    save_animation_eval(tsne_blended_02)
    save_animation_eval(tsne_cosine_03)
    save_animation_eval(tsne_cosine_02)
    save_animation_eval(tsne_p_5)
    save_animation_eval(tsne_p_10)
    save_animation_eval(tsne_p_30)
    save_animation_eval(tsne_p_50)

    logger.set_ani(run, tsne_animation)
    logger.set_ani(run, tsne_denoised)
    logger.set_ani(run, tsne_blended_03)
    logger.set_ani(run, tsne_blended_02)
    logger.set_ani(run, tsne_cosine_03)
    logger.set_ani(run, tsne_cosine_02)
    logger.set_ani(run, tsne_p_5)
    logger.set_ani(run, tsne_p_10)
    logger.set_ani(run, tsne_p_30)
    logger.set_ani(run, tsne_p_50)

    logger.save()

def evaluate_umap(run: Run, args):
    logger = CSVGridLogger("plots/results_umap.csv")

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.deprecation")

#    # Cosine vs Euclidean
#    umap_ani = generate_umap_animation(run, fit_basis='all_n')
#    umap_cosine = generate_umap_animation(run, fit_basis='all_n', metric='cosine')
#    umap_ani.save_as_gif()
#    umap_cosine.save_as_gif()
#    save_animation_eval(umap_ani)
#    save_animation_eval(umap_cosine)
#    logger.set_ani(run, umap_ani)
#    logger.set_ani(run, umap_cosine)
#
#    # Denoised
#    umap_ani_denoised = umap_ani.denoise()
#    umap_cosine_denoised = umap_cosine.denoise()
#    umap_ani_denoised.save_as_gif()
#    umap_cosine_denoised.save_as_gif()
#    save_animation_eval(umap_ani_denoised)
#    save_animation_eval(umap_cosine_denoised)
#    logger.set_ani(run, umap_ani_denoised)
#    logger.set_ani(run, umap_cosine_denoised)
#
#    # Neighbors
#    umap_neighbors_5 = generate_umap_animation(run, metric='cosine', n_neighbors=5)
#    umap_neighbors_10 = generate_umap_animation(run, metric='cosine', n_neighbors=10)
#    umap_neighbors_20 = generate_umap_animation(run, metric='cosine', n_neighbors=20)
#    umap_neighbors_30 = generate_umap_animation(run, metric='cosine', n_neighbors=30)
#    umap_neighbors_50 = generate_umap_animation(run, metric='cosine', n_neighbors=50)
#
#    umap_neighbors_5.save_as_gif()
#    umap_neighbors_10.save_as_gif()
#    umap_neighbors_20.save_as_gif()
#    umap_neighbors_30.save_as_gif()
#    umap_neighbors_50.save_as_gif()
#    save_animation_eval(umap_neighbors_5)
#    save_animation_eval(umap_neighbors_10)
#    save_animation_eval(umap_neighbors_20)
#    save_animation_eval(umap_neighbors_30)
#    save_animation_eval(umap_neighbors_50)
#
#    logger.set_ani(run, umap_neighbors_5)
#    logger.set_ani(run, umap_neighbors_10)
#    logger.set_ani(run, umap_neighbors_20)
#    logger.set_ani(run, umap_neighbors_30)
#    logger.set_ani(run, umap_neighbors_50)
#
#    logger.set_ani(run, umap_neighbors_5.denoise())
#    logger.set_ani(run, umap_neighbors_10.denoise())
#    logger.set_ani(run, umap_neighbors_20.denoise())
#    logger.set_ani(run, umap_neighbors_30.denoise())
#    logger.set_ani(run, umap_neighbors_50.denoise())
#
#    # Min Dist
#    umap_dist_001 = generate_umap_animation(run, metric='cosine', min_dist=0.001)
#    umap_dist_01 = generate_umap_animation(run, metric='cosine', min_dist=0.01)
#    umap_dist_2 = generate_umap_animation(run, metric='cosine', min_dist=0.2)
#    umap_dist_5 = generate_umap_animation(run, metric='cosine', min_dist=0.5)
#
#    umap_dist_001.save_as_gif()
#    umap_dist_01.save_as_gif()
#    umap_dist_2.save_as_gif()
#    umap_dist_5.save_as_gif()
#    save_animation_eval(umap_dist_001)
#    save_animation_eval(umap_dist_01)
#    save_animation_eval(umap_dist_2)
#    save_animation_eval(umap_dist_5)
#
#    logger.set_ani(run, umap_dist_001)
#    logger.set_ani(run, umap_dist_01)
#    logger.set_ani(run, umap_dist_2)
#    logger.set_ani(run, umap_dist_5)
#
#    logger.set_ani(run, umap_dist_001.denoise())
#    logger.set_ani(run, umap_dist_01.denoise())
#    logger.set_ani(run, umap_dist_2.denoise())
#    logger.set_ani(run, umap_dist_5.denoise())
#
#    # Educated Guess
#    umap_guess = generate_umap_animation(run, metric='cosine', min_dist=0.2, n_neighbors=20).denoise()
#    umap_guess.save_as_gif()
#    save_animation_eval(umap_guess)
#    logger.set_ani(run, umap_guess)

    # Online OptionGuess
    umap_guess = generate_umap_animation(run, metric='cosine', min_dist=0.2, n_neighbors=20, online=True)
    umap_guess.save_as_gif()
    save_animation_eval(umap_guess)
    logger.set_ani(run, umap_guess)

    umap_guess = umap_guess.denoise()
    umap_guess.save_as_gif()
    save_animation_eval(umap_guess)
    logger.set_ani(run, umap_guess)

    logger.save()

def evaluate_phate(run: Run, args):
    logger = CSVGridLogger("plots/results_phate.csv")

    m_phate_animation = generate_mphate_animation(run)
    m_phate_animation.save_as_gif()
    save_animation_eval(m_phate_animation)
    logger.set_ani(run, m_phate_animation)

    denoised = m_phate_animation.denoise()
    denoised.save_as_gif()
    save_animation_eval(denoised)
    logger.set_ani(run, denoised)

    for t in [10, 20, 30]:
        ani = generate_mphate_animation(run,
                                        title=f"M-PHATE (t={t})",
                                        t=t)
        ani.save_as_gif()
        save_animation_eval(ani)
        logger.set_ani(run, ani)
        denoised = ani.denoise()
        denoised.save_as_gif()
        save_animation_eval(denoised)
        logger.set_ani(run, denoised)

    for knn in [10, 20, 30]:
        ani = generate_mphate_animation(run,
                                        title=f"M-PHATE (interslice_knn={knn})",
                                        interslice_knn=knn)
        ani.save_as_gif()
        save_animation_eval(ani)
        logger.set_ani(run, ani)
        denoised = ani.denoise()
        denoised.save_as_gif()
        save_animation_eval(denoised)
        logger.set_ani(run, denoised)

    for gamma in [0.0, 0.05, 0.1, 0.2]:
        ani = generate_mphate_animation(run,
                                        title=f"M-PHATE (gamma={gamma})",
                                        gamma=gamma)
        ani.save_as_gif()
        save_animation_eval(ani)
        logger.set_ani(run, ani)
        denoised = ani.denoise()
        denoised.save_as_gif()
        save_animation_eval(denoised)
        logger.set_ani(run, denoised)

    # Educated Guess
    mphate_guess = generate_mphate_animation(run,
                                             title=f"M-PHATE (t=20, gamma=0, interslice_knn=10)",
                                             t=20, gamma=0, interslice_knn=10)
    mphate_guess.save_as_gif()
    save_animation_eval(mphate_guess)
    logger.set_ani(run, mphate_guess)

    mphate_guess = mphate_guess.denoise()
    mphate_guess.save_as_gif()
    save_animation_eval(mphate_guess)
    logger.set_ani(run, mphate_guess)

    logger.save()
    
def evaluate_best(run):
    from helper.visualization import compare_animations

    ani_pca_all = generate_pca_animation(run, fit_basis='all')
    #ani_pca_window = generate_pca_animation(run, fit_basis='window', window_size=16).denoise(do_cka_similarities=False)
    tsne_blended = generate_tsne_animation(run)
    umap_ani = generate_umap_animation(
        run,
        metric='cosine',
        n_neighbors=20,
        min_dist=0.2
    )
    mphate_ani = generate_mphate_animation(
        run,
        t='auto',
        gamma=0,
        interslice_knn=20
    )

    fig = compare_animations(
        animations=[
            ani_pca_all,
            #ani_pca_window,
            tsne_blended,
            umap_ani,
            mphate_ani
        ],
        custom_titles=[
            "PCA on all",
            #"PCA window denoised",
            "t-SNE",
            "UMAP",
            "M-PHATE"
        ],
        initial_datapoints=[0, 0.05, 0.1, 0.2, 0.5, 1],  # -> 3 independent row sliders
        figsize_per_plot=(3, 3),
        shared_axes=False,
        add_confusion_matrix=True,
        annotate_confusion_matrix=True,
    )

    fig.savefig(f"plots/comparison {run.run_id}.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_animation_eval(animation: Animation):
    fig = animation.eval_plot()
    fig.savefig(f"plots/{run.run_id}/{animation.title}.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    run, args = init_run()

    #general_plots(run, args)
    #evaluate_pca(run, args)
    #evaluate_umap(run, args)
    #evaluate_phate(run, args)
    #evaluate_tsne(run, args)

    evaluate_best(run)

