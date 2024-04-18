#python eval.py --checkpoint_dir checkpoint -b 64 -s 1 --fast_dev_run --run_name verify_me "checkpoint/test_me/epoch=01-step=17168-val_loss=nan.ckpt"
#python eval.py --checkpoint_dir checkpoint -b 64 -s 1 --fast_dev_run --run_name verify_lstme
#python eval.py --checkpoint_dir checkpoint -b 64 -s 1 --fast_dev_run --run_name verify_blstme
#python eval.py --checkpoint_dir checkpoint -b 64 -s 1 --fast_dev_run --run_name verify_blstmpme

python eval.py --checkpoint_dir checkpoint -b 64 -s 1  --run_name verify_me "checkpoint/verify_me_train/epoch=04-step=42920-val_loss=0.65.ckpt"
