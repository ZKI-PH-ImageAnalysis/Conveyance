# Conveyance

## Requirements

```console
python >= 3.9, torch >= 1.12.1, torchvision >= 0.13.1, numpy >= 1.23.1
```

## How to use

### Configs

Check '*.json' file in the config folder for each exeriment.
update DATA_DIR in main.pyto point to cifar-10, cifar-100

### Arguments

* gpu: GPU id
* seed: random seed
* config: config name
* noise_type: 
   on cifar-10:
         - 'asym_single' for asymmetric noise (directed graph)
         - 'column' for column noise
   on cifar-100:
         - 'asym' for asymmetric noise
         - 'block' for block noise

* noise_rate: noise rate (0.45 for asym, 0.6 otherwise)
* eval_freq: frequency of evaluation, default is 1
* tuning: use the tuning settings (90% of the original training set as training set and 10% as validation set)

### Example

Training conveyance on CIFAR-10 with 0.6 column noise:
```
python main.py \
--gpu 0 \
--seed 1 \
--config cifar10_conveyance \
--noise_type column \
--noise_rate 0.6 \
--eval_freq 10
```

### Reproduce results
To reproduce the results in the paper, run the followinf script:
```
chmod +x run_experiments.sh
./bin/run_experiments.sh
```
## Thanks

Moreover, parts of this repository is taken from [Ye et al.](https://github.com/Virusdoll/Active-Negative-Loss), [Ma et al.](https://github.com/HanxunH/Active-Passive-Losses) and [Zhou et al.](https://github.com/hitcszx/ALFs).