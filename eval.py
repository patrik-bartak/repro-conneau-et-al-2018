import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from src.dataset.dataloaders import create_dataloaders
from src.dataset.utils import get_eval_parser
from src.models.nliclassifier import NLIClassifier


def eval():
    """
    Evaluate a classifier on the NLI task with one of the four encoder types.
    """
    parser = get_eval_parser()
    args = parser.parse_args()

    args.run_name += "_test"

    if args.seed is not None:
        pl.seed_everything(args.seed)

    (dataloader_tst,), emb_vecs = create_dataloaders(args.batch_size, splits=["test"])

    # Load with embeddings because they are not saved in the model checkpoint
    model = NLIClassifier.load_from_checkpoint(
        args.checkpoint_path,
        strict=False,
        embedding_mat=emb_vecs,
    )

    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=args.checkpoint_dir, name=args.run_name),
        fast_dev_run=args.fast_dev_run,
    )
    trainer.test(model, dataloader_tst)


if __name__ == "__main__":
    eval()
