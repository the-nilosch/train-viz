import numpy as np
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import random

class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def init_dataset(
    dataset_name,
    max_length=256,
    batch_size=32,
    samples_per_class=None
):
    """
    Loads AG News, Yelp, or DBpedia from HuggingFace Datasets,
    automatically tokenizes using DistilBERT tokenizer,
    and returns train_loader, test_loader, and a subset_loader
    for embedding visualization.
    """
    dataset_name = dataset_name.lower()
    assert dataset_name in ['ag_news', 'yelp', 'dbpedia'], "Dataset must be one of ag_news, yelp, dbpedia"

    # === Load raw HuggingFace dataset ===
    if dataset_name == 'ag_news':
        raw = load_dataset("ag_news")
        num_classes = 4
    elif dataset_name == 'yelp':
        raw = load_dataset("yelp_review_full")
        num_classes = 5
    elif dataset_name == 'dbpedia':
        raw = load_dataset("dbpedia_14")
        num_classes = 14

    # --- Auto-load tokenizer ---
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # --- Tokenize function ---
    def tokenize_batch(batch):
        return tokenizer(
            batch['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )

    tokenized_train = raw['train'].map(tokenize_batch, batched=True)
    tokenized_test = raw['test'].map(tokenize_batch, batched=True)

    # === Build PyTorch datasets ===
    train_data = TextClassificationDataset(tokenized_train, tokenized_train['label'])
    eval_data = train_data  # Like vision (train_data used for sampling subset)
    test_data = TextClassificationDataset(tokenized_test, tokenized_test['label'])

    # === Get targets for balanced sampling ===
    subset_targets = np.array(tokenized_train['label'])

    # === Create loaders ===
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # === Build balanced subset loader ===
    selected_indices = []
    for class_id in range(num_classes):
        class_indices = np.where(subset_targets == class_id)[0]
        if len(class_indices) < samples_per_class:
            chosen = np.random.choice(class_indices, size=len(class_indices), replace=False)
        else:
            chosen = np.random.choice(class_indices, size=samples_per_class, replace=False)
        selected_indices.extend(chosen)

    subset = Subset(eval_data, selected_indices)
    subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    # === Print stats ===
    print(f"{len(train_data)} samples in train data")
    print(f"{len(test_data)} samples in test data")
    print(f"{len(subset)} samples in visualization subset")

    return train_loader, test_loader, subset_loader, num_classes


def init_nlp_model_for_dataset(
    dataset_name,
    model_name,
    hidden_dim=128,
    num_layers=1,
    dropout=0.2,
    tokenizer=None,
    device=None
):
    """
    Initializes an NLP model for a given dataset, mirroring the Vision-style init functions.

    Parameters:
    - dataset_name: str, e.g. "ag_news", "yelp", "dbpedia"
    - model_name: str, e.g. "bilstm", "textcnn", "distilbert"
    - hidden_dim, num_layers, dropout: model hyperparameters
    - tokenizer: only needed for DistilBERT
    - device: torch.device

    Returns:
    - model (on device)
    """

    # --- Define number of classes per dataset ---
    if dataset_name.lower() == "ag_news":
        num_classes = 4
    elif dataset_name.lower() == "yelp":
        num_classes = 5
    elif dataset_name.lower() == "dbpedia":
        num_classes = 14
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # --- Initialize model ---
    if model_name == "bilstm":
        import BiLSTMAttentionClassifier from models.bilstm_attention
        model = BiLSTMAttentionClassifier(
            vocab_size=...,  # Set to your vocab size (for word2vec it would be the vocab of the embeddings)
            embedding_dim=400,  # Word2Vec Twitter embeddings are 400-dimensional
            hidden_dim=400,  # BiLSTM layer has 400 hidden units (per direction)
            num_layers=1,  # Single-layer BiLSTM
            num_classes=6,  # 6 emotion classes in the original task (adjust for your dataset)
            dropout=dropout,  # Try 0.3–0.5, tune if needed
            pad_idx=your_pad_idx,  # Padding index, set according to your vocab
            use_attention=True
        )

    elif model_name == "textcnn":
        model = ParallelTextCNN(
            vocab_size=len(vocab),
            embedding_dim=300,
            num_classes=NUM_CLASSES,
            kernel_sizes=[2, 3, 4, 5],
            num_filters=512,
            dropout=0.5,
            pad_idx=PAD_IDX,
            pretrained_embeddings=your_tensor_or_None,
            freeze_embeddings=False,
            multi_label=False
        )
    elif model_name == "distilbert":
        if tokenizer is None:
            from transformers import DistilBertTokenizerFast
            tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        model = DistilBERTClassifier(
            num_classes=num_classes,
            tokenizer=tokenizer
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model.to(device)
