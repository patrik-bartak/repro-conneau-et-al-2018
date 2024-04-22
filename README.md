# learning-sentence-representations
Learning sentence representations from natural language inference data

Dataset used: https://huggingface.co/datasets/stanfordnlp/snli

## Model training

To train models, use `python train.py`. 

## Model evaluation

To test models on SNLI, use `python eval.py`. 

To run the senteval evaluation, use `python senteval.py`. This will run the `setup_senteval.sh` script, and then 
perform the senteval evaluation of all the models present in `./checkpoint/<ckpt_name>/final.ckpt`. 
