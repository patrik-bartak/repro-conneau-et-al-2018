python eval.py --checkpoint_dir checkpoint -b 64 -s 1 --fast_dev_run --run_name verify_me "checkpoint/<...>"
python eval.py --checkpoint_dir checkpoint -b 64 -s 1 --fast_dev_run --run_name verify_lstme "checkpoint/<...>"
python eval.py --checkpoint_dir checkpoint -b 64 -s 1 --fast_dev_run --run_name verify_blstme "checkpoint/<...>"
python eval.py --checkpoint_dir checkpoint -b 64 -s 1 --fast_dev_run --run_name verify_blstmpme "checkpoint/<...>"
