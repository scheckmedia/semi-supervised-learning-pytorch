#!/bin/bash

python train_cifar10.py -a=wideresnet --ckpt=ckpt --gpu=0,1 --boundary=0 &&
python train_cifar10.py -a=wideresnet --ckpt=ckpt --gpu=0,1 --boundary=1 &&
python train_cifar10.py -a=wideresnet --ckpt=ckpt --gpu=0,1 --boundary=2 &&
python train_cifar10.py -a=wideresnet --ckpt=ckpt --gpu=0,1 --boundary=3 &&
python train_cifar10.py -a=wideresnet --ckpt=ckpt --gpu=0,1 --boundary=4 &&
python train_cifar10.py -a=wideresnet --ckpt=ckpt --gpu=0,1 --boundary=5 &&
python train_cifar10.py -a=wideresnet --ckpt=ckpt --gpu=0,1 --boundary=6 &&
python train_cifar10.py -a=wideresnet --ckpt=ckpt --gpu=0,1 --boundary=7 &&
python train_cifar10.py -a=wideresnet --ckpt=ckpt --gpu=0,1 --boundary=8 &&
python train_cifar10.py -a=wideresnet --ckpt=ckpt --gpu=0,1 --boundary=9 &&


ls
