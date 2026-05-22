#!/bin/bash

# seeds
seeds=(1 2 3)

# cifar10_conveyance with column noise type
for seed in "${seeds[@]}"; do
  python main.py \
    --gpu 0 \
    --seed $seed \
    --config cifar10_conveyance \
    --noise_type column \
    --noise_rate 0.6 \
    --eval_freq 10
done

# cifar10_conveyance with asym_single noise type
for seed in "${seeds[@]}"; do
  python main.py \
    --gpu 0 \
    --seed $seed \
    --config cifar10_conveyance \
    --noise_type asym_single \
    --noise_rate 0.45 \
    --eval_freq 10
done

# cifar100_conveyance with block noise type
for seed in "${seeds[@]}"; do
  python main.py \
    --gpu 0 \
    --seed $seed \
    --config cifar100_conveyance \
    --noise_type block \
    --noise_rate 0.6 \
    --eval_freq 10
done

# cifar100_conveyance with asym noise type
for seed in "${seeds[@]}"; do
  python main.py \
    --gpu 0 \
    --seed $seed \
    --config cifar100_conveyance \
    --noise_type asym \
    --noise_rate 0.45 \
    --eval_freq 10
done