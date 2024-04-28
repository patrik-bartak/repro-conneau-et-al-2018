import os

import lightning as pl
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger

from src.dataset.dataloaders import create_dataloaders
from src.dataset.utils import (assert_args_are_valid, encoders, get_logger,
                               get_main_parser)
from src.models.nliclassifier import NLIClassifier


def train():
    """
    Train a classifier on the NLI task with one of the four encoder types.
    """
    parser = get_main_parser()
    args = parser.parse_args()

    args.run_name += "_train"

    assert_args_are_valid(args)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, args.run_name, "logs")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = get_logger(log_path)
    logger.info(f"Running with args {args}")

    if args.seed is not None:
        pl.seed_everything(args.seed)

    (dataloader_trn, dataloader_val), emb_vecs = create_dataloaders(
        args.batch_size, splits=["train", "validation"]
    )

    model_config = {
        "mlp_dims": args.mlp_dims,
        "encoder_type": encoders[args.encoder],
        "embedding_mat": emb_vecs,
        "lr": args.lr,
        "lr_decay": args.lr_decay,
        "encoder_in_size": args.encoder_in_size,
        "encoder_out_size": args.encoder_out_size,
    }

    model = NLIClassifier(**model_config)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    es_callback = EarlyStopping(
        monitor="lr-SGD", mode="min", patience=0, stopping_threshold=10e-5, verbose=True
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, args.run_name).__str__(),
        filename="{epoch:02d}-{step}-{val_acc:.2f}",
        save_on_train_epoch_end=True,
        enable_version_counter=True,
        every_n_epochs=5,
        save_top_k=-1,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        logger=TensorBoardLogger(save_dir=args.checkpoint_dir, name=args.run_name),
        fast_dev_run=args.fast_dev_run,
        val_check_interval=1.0,
        callbacks=[lr_monitor, es_callback, checkpoint_callback],
        devices=-1,
    )
    trainer.fit(model, dataloader_trn, dataloader_val)


if __name__ == "__main__":
    train()
