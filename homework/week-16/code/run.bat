@echo off
rem    Run this file on the command line of an environment that contains "python" in path
rem    For example, in the terminal of your IDE
rem    Or in the correct environment of your anaconda prompt

if "%1%"=="train" (
    set CUDA_VISIBLE_DEVICES=0 & python run.py train --train-src=./br_chr_data/train.br --train-tgt=./br_chr_data/train.chr --dev-src=./br_chr_data/dev.br --dev-tgt=./br_chr_data/dev.chr --vocab=vocab.json --cuda --lr=5e-4 --patience=1 --valid-niter=200 --batch-size=32 --dropout=.3
) else if "%1%"=="test" (
    set CUDA_VISIBLE_DEVICES=0 & python run.py decode model.bin ./br_chr_data/test.br ./br_chr_data/test.chr outputs/test_outputs.txt --cuda
) else if "%1%"=="train_local" (
    python run.py train --train-src=./br_chr_data/train.br --train-tgt=./br_chr_data/train.chr --dev-src=./br_chr_data/dev.br --dev-tgt=./br_chr_data/dev.chr --vocab=vocab.json --lr=5e-5
) else if "%1%"=="test_local" (
    python run.py decode model.bin ./br_chr_data/test.br ./br_chr_data/test.chr outputs/test_outputs.txt
) else if "%1%"=="vocab" (
    python vocab.py --train-src=./br_chr_data/train.br --train-tgt=./br_chr_data/train.chr vocab.json
) else (
    echo Invalid Option Selected
)
