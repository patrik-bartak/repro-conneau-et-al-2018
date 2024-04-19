python train.py --encoder me --mlp_dims 1200 512 3 --eo 300 --run_name real_me -b 64 -s 1 -e 40
python train.py --encoder lstme --mlp_dims 4096 512 3 --eo 1024 --run_name real_lstme -b 64 -s 1 -e 40
python train.py --encoder blstme --mlp_dims 8192 512 3 --eo 2048 --run_name real_blstme -b 64 -s 1 -e 40
python train.py --encoder blstmpme --mlp_dims 8192 512 3 --eo 2048 --run_name real_blstmpme -b 64 -s 1 -e 40
