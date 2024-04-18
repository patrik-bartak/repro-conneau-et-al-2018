import argparse
import logging

from src.encoders.bilstm_encoder import BiLSTMEncoder
from src.encoders.bilstm_max_pool import BiLSTMMaxPoolEncoder
from src.encoders.lstm_encoder import LSTMEncoder
from src.encoders.mean_encoder import MeanEncoder

dataset_splits = ["train", "validation", "test"]
dataset_feats = ["premise", "hypothesis", "label"]

encoders = {
    "me": MeanEncoder,
    "lstme": LSTMEncoder,
    "blstme": BiLSTMEncoder,
    "blstmpme": BiLSTMMaxPoolEncoder,
}

encoders_with_hidden_size = {
    "lstme": LSTMEncoder,
    "blstme": BiLSTMEncoder,
    "blstmpme": BiLSTMMaxPoolEncoder,
}


def get_splits(dataset):
    return [dataset[split] for split in dataset_splits]


def get_feats(split, include_labels=True):
    if include_labels:
        return [split[feat] for feat in dataset_feats]
    else:
        return [split[feat] for feat in dataset_feats if feat != "label"]


def flatten_nested_lists(lists):
    return [item for lst in lists for item in lst]


def get_logger(log_path, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    log_formatter = logging.Formatter(fmt="[%(levelname)s] %(message)s")
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(log_path, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(log_formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)
    return logger


def get_eval_parser():
    parser = argparse.ArgumentParser(description="NLI Evaluation")

    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="path to save checkpoints and logs",
    )

    parser.add_argument("-d", "--data", default="data", help="path to dataset")
    parser.add_argument(
        "--fast_dev_run",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="runs ax single train/val/test batch to ensure there are no errors",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoint",
        type=str,
        help="path to save checkpoints and logs",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="name of run for logging",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=64,
        type=int,
        help="mini-batch size",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        help="print frequency",
    )
    parser.add_argument(
        "-s", "--seed", default=None, type=int, help="seed for initializing training. "
    )
    return parser


def get_main_parser():
    parser = argparse.ArgumentParser(description="NLI Training")

    # Required
    parser.add_argument(
        "--encoder",
        choices=encoders.keys(),
        required=True,
        help="type of sentence encoder to use",
    )
    parser.add_argument(
        "--mlp_dims",
        required=True,
        type=int,
        nargs="+",
        help="mlp dimensions to use on encoded sentences",
    )
    parser.add_argument("--run_name", type=str, required=True, help="name of the run")
    parser.add_argument(
        "--eo",
        "--encoder_out_size",
        type=int,
        required=True,
        help="size of encoder output representation",
        dest="encoder_out_size",
    )

    # Optional
    parser.add_argument(
        "--ei",
        "--encoder_in_size",
        type=int,
        default=300,
        help="size of encoder input representation",
        dest="encoder_in_size",
    )
    parser.add_argument("-d", "--data", default="data", help="path to dataset")
    parser.add_argument(
        "-e", "--epochs", default=1, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "--max_steps", default=-1, type=int, help="number of maximum steps to run"
    )
    parser.add_argument(
        "--fast_dev_run",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="runs ax single train/val/test batch to ensure there are no errors",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoint",
        type=str,
        help="path to save checkpoints and logs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=64,
        type=int,
        help="mini-batch size",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum for adam")
    parser.add_argument(
        "--lrd",
        "--lr-decay",
        default=0.99,
        type=float,
        help="lr decay",
        dest="lr_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        help="print frequency",
    )
    parser.add_argument(
        "-s", "--seed", default=None, type=int, help="seed for initializing training. "
    )
    return parser


def assert_args_are_valid(args):
    assert args.mlp_dims[-1] == 3, "Final mlp dim must be 3"

    if args.encoder == "me":
        # Should be 4 times the glove embedding size due to concatenation
        assert args.mlp_dims[0] == args.encoder_in_size * 4
        assert args.encoder_in_size == args.encoder_out_size == 300

    if args.encoder in encoders_with_hidden_size:
        assert (
            args.encoder_out_size
        ), "Encoder out size must be set for LSTM-based models"
        # Enforce correct dimensions of encoder to mlp
        assert args.mlp_dims[0] == args.encoder_out_size * 4
