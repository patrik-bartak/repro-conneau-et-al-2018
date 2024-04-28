import logging
import os
import pickle
from typing import Tuple

import torch.utils.data
from datasets import load_dataset, load_from_disk
from spacy.lang.en import English
from torch.utils.data import DataLoader

from src.dataset.snli_dataset import (
    SNLIDataset, collate_nli, get_aligned_glove_embeddings_from_unique_tokens,
    get_unique_tokens, process_data)
from src.dataset.utils import dataset_splits, get_feats


def get_processed_data(dataset_path=os.path.join("data", "processed")):
    """
    Read in the processed data based on a path.

    :param dataset_path: Path to the dataset.
    :return: The processed data.
    """
    # Data
    dataset = load_dataset("stanfordnlp/snli")
    # Load embeddings
    tok = English().tokenizer
    if os.path.exists(dataset_path):
        processed_data = load_from_disk(dataset_path)
    else:
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        processed_data = process_data(dataset, tok)
        processed_data.save_to_disk(dataset_path)
    return processed_data


def get_embeddings_for_data(dataset_path=os.path.join("data", "processed")):
    """
    Read or download GLoVe embeddings for a data path.

    :param dataset_path: Path to dataset.
    :return: Embedding vocab and vectors.
    """
    data = get_processed_data(dataset_path)
    embedding_path = os.path.join(dataset_path, "embeddings.pickle")
    if os.path.exists(embedding_path):
        with open(embedding_path, "rb") as f:
            emb_vocab, emb_vecs = pickle.load(f)
    else:
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        tokens = get_unique_tokens(data)
        emb_vocab, emb_vecs = get_aligned_glove_embeddings_from_unique_tokens(tokens)
        with open(embedding_path, "wb") as f:
            pickle.dump((emb_vocab, emb_vecs), f)
    return emb_vocab, emb_vecs


def create_dataloaders(
    batch_size: int,
    dataset_path: str = os.path.join("data", "processed"),
    splits: list[str] = dataset_splits,
    data_fraction: int = 1,
) -> Tuple[list[torch.utils.data.DataLoader], torch.Tensor]:
    """
    Get the dataloaders and embedding vectors based on a path.

    :param batch_size: Batch size for the loader.
    :param dataset_path: Path to read data from.
    :param splits: Splits to include in the loaders.
    :param data_fraction: Fraction of data to load.
    :return: The dataloaders and embedding vectors
    """
    processed_data = get_processed_data(dataset_path)
    emb_vocab, emb_vecs = get_embeddings_for_data(dataset_path)
    dataloaders = []

    for split in splits:
        split_dataset = SNLIDataset(
            *get_feats(processed_data[split]), emb_vocab, frac=data_fraction
        )
        split_dataloader = DataLoader(
            split_dataset,
            batch_size=batch_size,
            shuffle=True if split == "train" else False,
            collate_fn=collate_nli,
            num_workers=8,
            persistent_workers=True,
        )
        logging.info(
            f"Dataloader {split} prepared with {len(split_dataset)}/{len(split_dataloader)}/{batch_size} (#row/#batch/b_size)"
        )
        dataloaders.append(split_dataloader)

    return dataloaders, emb_vecs
