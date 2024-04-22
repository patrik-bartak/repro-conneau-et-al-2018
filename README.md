# Learning sentence representations
Learning sentence representations from natural language inference data

Dataset used: https://huggingface.co/datasets/stanfordnlp/snli

# Setup

To set up the requirements, run `conda env create -f environment.yaml`. Then activate 
the environment using `conda activate atcs`. 

# Structure

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
An example of how this file was used can be seen in `train_all_models.sh`. 

# Model evaluation

To test models on SNLI, use `python eval.py`. Use the `--help` flag to learn more about possible arguments. 
An example of how this file was used can be seen in `eval_all_models.sh`. 

To run the senteval evaluation, use `python senteval.py`. This will run the `setup_senteval.sh` script, and then 
perform the senteval evaluation of all the models present in `./checkpoint/<ckpt_name>/final.ckpt`. 
