set -e

git clone https://github.com/facebookresearch/SentEval.git
cd SentEval/
python setup.py install

cd data/downstream/
./get_transfer_data.bash
