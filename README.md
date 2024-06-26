# Learning sentence representations
Learning sentence representations from natural language inference data using the [SNLI dataset](https://huggingface.co/datasets/stanfordnlp/snli). 

# Setup

To set up the requirements, run `conda env create -f environment.yaml`. Then activate 
the environment using `conda activate atcs`. 

Model checkpoints (and Tensorboard logs) in the form of a `checkpoint` folder are available at [this link](https://drive.google.com/drive/folders/1r1Rz8RfBsE_tOXzfJiaPJ4MuiPYB6ee3?usp=share_link). 

Tensorboard logs are in: 
  - `checkpoint/real_me_train/version_0/`. 
  - `checkpoint/real_lstme_train/version_0/`. 
  - `checkpoint/real_blstme_train/version_0/`. 
  - `checkpoint/real_blstmpme_train/version_0/`. 

The `demonstration.ipynb`, `eval.py` and `run_senteval.py` expect to find the checkpoints at the following locations, 
but you can easily unzip the embeddings from the link above and the files structure should be correct: 
- `checkpoint/real_me_train/final.ckpt`
- `checkpoint/real_lstme_train/final.ckpt`
- `checkpoint/real_blstme_train/final.ckpt`
- `checkpoint/real_blstmpme_train/final.ckpt`

# Structure

- The demo notebook is located in `notebooks/demonstration.ipynb`. 
- The report is located in `report.md`. 
- Senteval results are in `senteval_results/`, 
- Source code is in `src`. 
- Notebooks are in `notebooks`. 
- Job files are in `jobfiles`.

The following folders may be created when running scripts:

- When training models, metrics and checkpoints are saved to `checkpoint`. 
- Data and saved embeddings are downloaded to `data`. 
- Raw embeddings are downloaded to `.vector_cache`. 
- The senteval repo is downloaded to `SentEval`. 

# Model training

To train a model, use `python train.py`. Use the `--help` flag to learn more about possible arguments. 
An example of how this file was used can be seen in `train_all_models.sh`. You can also run `verify_train_all_models.sh`, 
which runs a single train/val step to validate the framework. 

# Model evaluation

To test models on SNLI, use `python eval.py`. Use the `--help` flag to learn more about possible arguments. 
An example of how this file was used can be seen in `eval_all_models.sh`. You can also run `verify_eval_all_models.sh`, 
which runs a single test step to validate the framework. 

To run the senteval evaluation, use `python senteval.py`. This will run the `setup_senteval.sh` script, and then 
perform the senteval evaluation of all the models present in `./checkpoint/<ckpt_name>/final.ckpt`. 
