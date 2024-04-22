from __future__ import absolute_import, division, unicode_literals

import json
import logging
import os
import sys

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
OUTPUT_DIR = "./senteval_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        # "STS14",
    ]
    return se.eval(transfer_tasks)


model_abbreviations = [
    "me",
    "lstme",
    "blstme",
    "blstmpme",
]

ckpt_paths = [
    (
        os.path.join(
            "checkpoint",
            f"real_{abbrv}_train",
            "final.ckpt",
        ),
        abbrv,
    )
    for abbrv in model_abbreviations
]


def write_result_to_file(file_name, results_dict, print_res=False):
    if print_res:
        print(results_dict)
    with open(os.path.join(OUTPUT_DIR, file_name), "w") as f:
        json.dump(results_dict, f, indent=4)


def main():
    results = []
    for ckpt_path, model_abbrv in ckpt_paths:
        print(f"Running senteval on checkpoint {ckpt_path}")
        res = run_senteval_on_checkpoint(ckpt_path)

        write_result_to_file(
            f"senteval_results_{model_abbrv}.json", res, print_res=True
        )
        results.append(res)

    write_result_to_file("senteval_results_all.json", results)
    print(f"Senteval finished")


if __name__ == "__main__":
    main()
