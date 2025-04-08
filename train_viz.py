import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from IPython.display import clear_output
import logging


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
