python train.py --encoder me --mlp_dims 1200 512 3 --eo 300 --run_name verify_me_works -b 64 -s 1 --fast_dev_run
python train.py --encoder lstme --mlp_dims 4096 512 3 --eo 1024 --run_name verify_lstme_works -b 64 -s 1 --fast_dev_run
python train.py --encoder blstme --mlp_dims 8192 512 3 --eo 2048 --run_name verify_blstme_works -b 64 -s 1 --fast_dev_run
python train.py --encoder blstmpme --mlp_dims 8192 512 3 --eo 2048 --run_name verify_blstmpme_works -b 64 -s 1 --fast_dev_run
