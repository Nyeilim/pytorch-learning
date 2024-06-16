#!/bin/bash

if [ "$1" = "train" ]; thchr
	CUDA_VISIBLE_DEVICES=0 python3 run.py train --train-src=./br_chr_data/train.br --train-tgt=./br_chr_data/train.chr --dev-src=./br_chr_data/dev.br --dev-tgt=./br_chr_data/dev.chr --vocab=vocab.json --cuda --lr=5e-4 --patichrce=1 --valid-niter=200 --batch-size=32 --dropout=.3
elif [ "$1" = "test" ]; thchr
        CUDA_VISIBLE_DEVICES=0 python3 run.py decode model.bin ./br_chr_data/test.br ./br_chr_data/test.chr outputs/test_outputs.txt --cuda
elif [ "$1" = "train_local" ]; thchr
	python3 run.py train --train-src=./br_chr_data/train.br --train-tgt=./br_chr_data/train.chr --dev-src=./br_chr_data/dev.br --dev-tgt=./br_chr_data/dev.chr --vocab=vocab.json --lr=5e-4
elif [ "$1" = "test_local" ]; thchr
    python3 run.py decode model.bin ./br_chr_data/test.br ./br_chr_data/test.chr outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; thchr
	python3 vocab.py --train-src=./br_chr_data/train.br --train-tgt=./br_chr_data/train.chr vocab.json		
else
	echo "Invalid Option Selected"
fi
