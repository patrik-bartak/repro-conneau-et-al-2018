python eval.py --checkpoint_dir checkpoint -b 64 -s 1 --run_name real_me "checkpoint/real_me_train/final.ckpt"
python eval.py --checkpoint_dir checkpoint -b 64 -s 1 --run_name real_lstme "checkpoint/real_lstme_train/final.ckpt"
python eval.py --checkpoint_dir checkpoint -b 64 -s 1 --run_name real_blstme "checkpoint/real_blstme_train/final.ckpt"
python eval.py --checkpoint_dir checkpoint -b 64 -s 1 --run_name real_blstmpme "checkpoint/real_blstmpme_train/final.ckpt"
