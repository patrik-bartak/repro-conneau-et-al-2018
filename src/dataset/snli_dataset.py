import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from torchtext.vocab import GloVe, Vocab, build_vocab_from_iterator

from src.dataset.utils import get_feats, get_splits


def tokenize_sentence(sentence, tokenizer):
    return [token.text for token in tokenizer(sentence.lower())]


def process_data(dataset, tokenizer):
    feats = ["premise", "hypothesis"]

    token_lens = []

    def is_valid_label(row):
        return row["label"] in {0, 1, 2}

    def tokenize_row(row):
        for feat in feats:
            row[feat] = tokenize_sentence(row[feat], tokenizer)
            token_lens.append(len(row[feat]))
        return row

    logging.info("Filtering out invalid labels")
    filtered_dataset = dataset.filter(is_valid_label)
    logging.info("Tokenizing dataset splits")
    tokenized_dataset = filtered_dataset.map(tokenize_row)

    mean_token_len = np.mean(token_lens)
    std_token_len = np.std(token_lens)
    logging.info(f"Processed data with mean token length {mean_token_len} and std {std_token_len}")

    return tokenized_dataset


def get_unique_tokens(dataset):
    tokens = {
        token
        for split in get_splits(dataset)
        for feat_split in get_feats(split, include_labels=False)
        for data_instance in feat_split
        for token in data_instance
    }
    logging.info(f"Found {len(tokens)} unique tokens in dataset")
    return tokens


def get_aligned_glove_embeddings_from_unique_tokens(tokens):
    """
    Get the glove embeddings necessary for the train/test/val dataset
    :return:
    """
    custom_vocab = build_vocab_from_iterator(
        [tokens], specials=["<unk>"], special_first=True
    )
    # Ensure <unk> is default
    custom_vocab.set_default_index(0)
    logging.info(f"Created custom vocab of size {len(custom_vocab.get_itos())}")

    glove_embeddings = GloVe(name="840B", dim=300)
    logging.info(f"Loaded GloVe embeddings of size {len(glove_embeddings)}")
    # Get only the ones we need
    # glove_embeddings.get_vecs_by_tokens(custom_vocab.get_itos())
    custom_embs = glove_embeddings.get_vecs_by_tokens(custom_vocab.get_itos())
    logging.info(f"Selected {custom_embs.shape[0]} embeddings")
    # tok to idx, idx to vec
    return custom_vocab, custom_embs


def collate_nli(dset_items):
    token_idxs_p, token_idxs_h, lengths_p, lengths_h, labels = zip(*dset_items)

    padded_token_idxs_p = torch.nn.utils.rnn.pad_sequence(
        token_idxs_p, batch_first=True
    )
    padded_token_idxs_h = torch.nn.utils.rnn.pad_sequence(
        token_idxs_h, batch_first=True
    )
    lengths_p = torch.stack(lengths_p)
    lengths_h = torch.stack(lengths_h)
    labels = torch.stack(labels)

    return padded_token_idxs_p, padded_token_idxs_h, lengths_p, lengths_h, labels


def senteval_collate(tok_batch, emb_vocab):
    idxs_batch = []
    len_batch = []
    for tokens in tok_batch:
        # Some samples are empty, so make them nonempty
        tokens = ["."] if tokens == [] else tokens
        idxs = emb_vocab(tokens)
        l = len(idxs)
        idxs_batch.append(torch.tensor(idxs, dtype=torch.long))
        len_batch.append(torch.tensor(l, dtype=torch.int64))

    padded_token_idxs = torch.nn.utils.rnn.pad_sequence(idxs_batch, batch_first=True)
    lengths = torch.stack(len_batch)

    return padded_token_idxs, lengths


def str_to_idxs(sent, tokenizer, emb_vocab):
    sent_tokens = tokenize_sentence(sent, tokenizer)
    indices = emb_vocab(sent_tokens)
    idxs_tensor = torch.tensor([indices], dtype=torch.long)
    len_tensor = torch.tensor([len(sent_tokens)], dtype=torch.int64)
    return idxs_tensor, len_tensor


label_map = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}


class SNLIDataset(Dataset):

    def __init__(
        self,
        premises: list[list[str]],
        hypotheses: list[list[str]],
        labels,
        embedding_vocab: Vocab,
        frac: float = 1.0,
    ):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.embedding_vocab = embedding_vocab
        self.frac = frac

    def __len__(self):
        return int(len(self.premises) * self.frac)

    def __getitem__(self, index):
        toks_p = self.premises[index]
        toks_h = self.hypotheses[index]
        label = self.labels[index]

        idxs_p = self.toks_to_idxs(toks_p)
        idxs_h = self.toks_to_idxs(toks_h)
        label = torch.tensor(label, dtype=torch.long)
        length_p = torch.tensor(idxs_p.shape[0], dtype=torch.long)
        length_h = torch.tensor(idxs_h.shape[0], dtype=torch.long)

        return idxs_p, idxs_h, length_p, length_h, label

    def toks_to_idxs(self, tokens):
        return torch.tensor(self.embedding_vocab(tokens), dtype=torch.long)
