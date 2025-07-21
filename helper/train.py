import math
import os

import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from IPython.display import clear_output
import logging
from sklearn.metrics import confusion_matrix

import helper.plots as plots
import helper.visualization as visualization


def train_model_with_embedding_tracking(
        model, train_loader, test_loader, subset_loader, device, num_classes,
        epochs=10, learning_rate=0.001, embedding_records_per_epoch=10, average_window_size=30,
        track_gradients=True, track_embedding_drift=True, track_cosine_similarity=False, track_scheduled_lr=False,
        track_pca=False, track_confusion_matrix=False, early_stopping=True, patience=4, weight_decay=0.05,
        optimizer=None, scheduler=None, use_sam=False, rho=0.02, save_model_weights_each_epoch=False
):
    assert model.__class__.__name__ in ['ViT', 'CNN', 'MLP', 'ResNet',
                                        'DenseNet'], "Model must be ViT, CNN, ResNet, DenseNet or MLP"
    optimizer, scheduler, criterion, train_config = _setup_training(model, learning_rate, epochs, weight_decay,
                                                                    optimizer=optimizer, scheduler=scheduler,
                                                                    use_sam=use_sam, rho=rho)

    # Initialize lists for performance tracking
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    scheduler_history = []
    val_confusion_matrices = []
    val_distributions = []

    # Loss Landscape ID
    run_id = None

    # Initialize lists for embedding snapshot
    num_batches = len(train_loader)
    embedding_batch_interval = math.ceil(num_batches / embedding_records_per_epoch)
    print(
        f"{num_batches} Batches, {embedding_records_per_epoch} Records per Epoch, Resulting Batch interval: {embedding_batch_interval}")
    embedding_snapshots, embedding_snapshot_labels, embedding_indices = [], [], []
    embedding_counter = 0

    # Initialize lists for gradient tracking
    gradient_norms, max_gradients, grad_param_ratios, gradient_indices = [], [], [], []
    gradient_counter = 0  # will track absolute batch index for x-axis

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

            if use_sam:
                def closure():
                    optimizer.zero_grad()
                    out = model(data)
                    loss = criterion(out, target)
                    loss.backward()
                    return loss

                # SAM runs closure twice inside step
                loss = optimizer.step(closure)

                # After SAM update, need fresh output for metrics
                with torch.no_grad():
                    output = model(data)

            else:
                # Vanilla one-step
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # 4) Gradient tracking (once per batch)
            if track_gradients and (batch_idx % int(embedding_batch_interval / 2) == 0):
                grad_norm, max_grad, grad_ratio = _track_gradients(model)
                gradient_norms.append(grad_norm)
                max_gradients.append(max_grad)
                grad_param_ratios.append(grad_ratio)
                gradient_indices.append(gradient_counter)
                gradient_counter += 1

            # 5) Embedding snapshots & drift
            if batch_idx % embedding_batch_interval == 0:
                model.eval()
                with torch.no_grad():
                    batch_embeddings, batch_labels = [], []
                    for data_sub, target_sub in subset_loader:
                        data_sub = data_sub.to(device)
                        _, emb = model(data_sub, return_embedding=True)
                        batch_embeddings.append(emb.cpu().numpy())
                        batch_labels.append(target_sub.numpy())
                    snapshot = np.concatenate(batch_embeddings, axis=0)
                    labels = np.concatenate(batch_labels, axis=0)
                embedding_snapshots.append(snapshot)
                embedding_snapshot_labels.append(labels)

                if track_embedding_drift:
                    embedding_drifts = visualization.calculate_embedding_drift(embedding_snapshots)
                model.train()
                embedding_indices.append(embedding_counter)
                embedding_counter += 1

                # Metrics (same for both)
                _, preds = torch.max(output, dim=1)
                correct_train += (preds == target).sum().item()
                total_train += target.size(0)
                epoch_train_loss += loss.item()

        # === Epoch-wise accuracy ===
        train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # === Validation phase ===
        model.eval()
        all_preds = []
        all_targets = []
        all_distributions = []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data, return_embedding=True)

                # scale loss by batch size
                batch_size = data.size(0)
                loss = criterion(output, target).item() * batch_size
                total_loss += loss
                total_samples += batch_size

                # predictions
                probs = softmax(output, dim=1)
                _, preds = torch.max(output, 1)

                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(target.cpu().tolist())
                all_distributions.extend(probs.cpu().tolist())

        # now divide by total number of samples
        val_loss = total_loss / total_samples

        # standard accuracy
        val_acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)

        # confusion matrix
        cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
        val_confusion_matrices.append(cm)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_distributions.append(np.array(all_distributions))

        if scheduler is not None:
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            scheduler_history.append(scheduler.get_last_lr()[-1])
        else:
            scheduler_history.append(learning_rate)

        # Loss Landscape: Save flattened model weights
        if save_model_weights_each_epoch:
            run_id = _save_model(model, epoch, run_id)

        # Live plot update
        fig, axs = _live_plot_update(num_figures=num_figures)
        plots.plot_loss_accuracy(axs[0], epoch, epochs, train_losses, val_losses, train_accuracies, val_accuracies)

        pos = 1
        if track_gradients:
            plots.plot_gradients(axs[pos], gradient_indices, gradient_norms, max_gradients, grad_param_ratios,
                            average_window_size)
            pos += 1
        if track_scheduled_lr and scheduler is not None:
            plots.plot_scheduled_lr(axs[pos], scheduler_history)
            pos += 1
        if track_embedding_drift:
            plots.plot_embedding_drift(axs[pos], embedding_drifts)
            pos += 1
        if track_confusion_matrix:
            plots.plot_confusion_matrix(axs[pos], val_confusion_matrices[-1], classes=list(range(num_classes)))
            pos += 1
        if track_pca:
            plots.plot_pca(axs[pos], embedding_snapshots, embedding_snapshot_labels, embedding_records_per_epoch,
                      num_classes=num_classes)
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
            log_line = f"Early stopping triggered at epoch {epoch + 1}"
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
        'val_confusion_matrices': val_confusion_matrices,
        'val_distributions': val_distributions,
        'subset_embeddings': embedding_snapshots,
        'subset_labels': embedding_snapshot_labels,
        'embedding_drifts': embedding_drifts,
        'gradient_norms': gradient_norms,
        'max_gradients': max_gradients,
        'grad_param_ratios': grad_param_ratios,
        'scheduler_history': scheduler_history,
        'll_flattened_weights_dir': run_id,
        'model_info': repr(model),
        'train_config': train_config
    }


def _save_model(model, epoch, next_run_id=None):
    """ Loss Landscape: Save flattened model weights """
    import os, re

    if next_run_id is None:
        # list all entries in trainings/
        entries = os.listdir('trainings')

        # extract numbers from names like run-0001, run-0002, …
        nums = [
            int(m.group(1))
            for e in entries
            if (m := re.match(r'run-(\d+)-.*$', e))
        ]

        # determine next index
        next_idx = max(nums) + 1 if nums else 1

        # format and print
        next_run_id = f"run-{next_idx:04d}-{model.__class__.__name__}"
        print(f'Saving model to trainings/{next_run_id}')

    dir_path = f'trainings/{next_run_id}/'
    os.makedirs(dir_path, exist_ok=True)

    weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu()
    torch.save(weights, os.path.join(dir_path, f'model-{epoch}.pt'))

    return next_run_id


def _setup_training(model, learning_rate, epochs, weight_decay, use_sam=False, optimizer=None, scheduler=None,
                    rho=0.02):
    supported = ['ViT', 'CNN', 'MLP', 'ResNet', 'DenseNet']
    model_name = model.__class__.__name__
    assert model_name in supported, f"Model must be one of: {supported}"

    # 1) Select base optimizer if not provided
    if optimizer is None:
        if model_name == 'MLP':
            base_opt = Adam(model.parameters(),
                            lr=learning_rate)
        elif model_name == 'ViT':
            base_opt = AdamW(model.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay or 1e-2)
        else:  # CNN, ResNet, DenseNet
            base_opt = SGD(model.parameters(),
                           lr=learning_rate,
                           momentum=0.9,
                           weight_decay=weight_decay or 1e-4)
    else:
        base_opt = optimizer

    # 2) Wrap in SAM if requested
    if use_sam:
        import sys
        # ensure vendored sam/ is on path
        this_dir = os.path.dirname(os.path.abspath(__file__))
        sam_path = os.path.join(this_dir, "sam")
        sys.path.insert(0, sam_path)

        # Import the PyTorch‐SAMSGD wrapper
        try:
            from sam.sam import SAMSGD
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(
                f"Could not import SAMSGD from {sam_path!r}: {e}\n"
                "Make sure you've cloned the moskomule/sam.pytorch repo into `sam/`."
            )

        # Only valid if base_opt is SGD
        #if not isinstance(base_opt, SGD):
        #    raise ValueError("SAMSGD can only wrap torch.optim.SGD")
        optimizer = SAMSGD(
            model.parameters(),
            lr=learning_rate,
            momentum=base_opt.defaults.get('momentum', 0.0),
            dampening=base_opt.defaults.get('dampening', 0.0),
            weight_decay=base_opt.defaults.get('weight_decay', 0.0),
            nesterov=base_opt.defaults.get('nesterov', False),
            rho=rho
        )
    else:
        optimizer = base_opt

    # 3) Select scheduler if not provided
    if scheduler is None:
        if model_name == 'MLP':
            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        else:
            # For ViT you may add warmup externally; here we use plain cosine
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # 4) Loss
    criterion = CrossEntropyLoss()

    # 5) Build config string and print summary
    opt_name = optimizer.__class__.__name__
    sched_name = scheduler.__class__.__name__
    config_string = (
        f"{model_name}|opt={opt_name}|lr={learning_rate}"
        f"|wd={weight_decay}|sam={use_sam}"
    )
    print(f"[Config] model={model_name}, optimizer={opt_name}, "
          f"scheduler={sched_name}, use_SAM={use_sam}")

    return optimizer, scheduler, criterion, config_string


def _live_plot_update(num_figures=1, ncols=2):
    plt.close('all')
    clear_output(wait=True)
    nrows = math.ceil(num_figures / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
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

