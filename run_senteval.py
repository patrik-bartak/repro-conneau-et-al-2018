from __future__ import absolute_import, division, unicode_literals

import json
import logging
import os
import sys

from src.dataset.dataloaders import get_embeddings_for_data
from src.dataset.snli_dataset import (
    get_aligned_glove_embeddings_from_unique_tokens, senteval_collate)
from src.dataset.utils import flatten_nested_lists
from src.models.nliclassifier import NLIClassifier

logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)

os.system("bash setup_senteval.sh")

# Set PATHs
PATH_TO_SENTEVAL = "./SentEval"
PATH_TO_DATA = "./SentEval/data"
PATH_TO_EMBEDDINGS = "./data/processed"
PATH_TO_CHECKPOINTS = "./checkpoint"
OUTPUT_PATH = "./senteval_results.json"

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def run_senteval_on_checkpoint(ckpt_path):
    def prepare(params, samples):
        # Select embeddings based on the vocabulary of this task, rather than the original embeddings
        unique_tokens = set(flatten_nested_lists(samples))
        params.emb_vocab, params.emb_vecs = (
            get_aligned_glove_embeddings_from_unique_tokens(unique_tokens)
        )
        # params.emb_vocab, params.emb_vecs = get_embeddings_for_data(
        #     dataset_path=PATH_TO_EMBEDDINGS
        # )

        full_model = (
            NLIClassifier.load_from_checkpoint(
                ckpt_path,
                strict=False,
                embedding_mat=params.emb_vecs,
            )
            .cpu()
            .eval()
        )
        params.nn_embedding = full_model.embedding
        params.encoder = full_model.encoder

    def batcher(params, batch):
        padded_token_idxs, lengths = senteval_collate(batch, params.emb_vocab)
        padded_token_embeddings = params.nn_embedding(padded_token_idxs)
        batch_embeddings = params.encoder(padded_token_embeddings, lengths)
        return batch_embeddings.cpu().detach().numpy()

    params_senteval = {"task_path": PATH_TO_DATA, "usepytorch": False, "kfold": 10}

    se = senteval.SE(params_senteval, batcher, prepare)

    transfer_tasks = [
        "MR",
        "CR",
        "MPQA",
        "SUBJ",
        "SST2",
        "TREC",
        "MRPC",
        "SICKEntailment",
        "STS14",
    ]
    return se.eval(transfer_tasks)


model_abbreviations = [
    "me",
    "lstme",
    "blstme",
    "blstmpme",
]

ckpt_paths = [
    os.path.join(
        "checkpoint",
        f"real_{abbrv}_train",
        "final.ckpt",
    )
    for abbrv in model_abbreviations
]


def main():
    results = []
    for ckpt_path in ckpt_paths:
        print(f"Running senteval on checkpoint {ckpt_path}")
        res = run_senteval_on_checkpoint(ckpt_path)
        print(res)
        results.append(res)

    with open(OUTPUT_PATH, "w") as json_res:
        json.dump(results, json_res, indent=4)


if __name__ == "__main__":
    main()