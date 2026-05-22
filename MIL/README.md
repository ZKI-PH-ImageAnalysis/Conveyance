# Conveyance MIL

**Conveyance MIL** is a multiple-instance learning framework built around the **Conveyance** instance loss; a probabilistic loss that encodes the asymmetric nature of the noise found in MIL datasets; positive bags contain negative patches but not the other way around.

The codebase supports classic MIL benchmark datasets (Musk1/2, Fox, Tiger, Elephant) and the Camelyon16 whole-slide image dataset.

---

## Environment

```bash
conda activate mil
```

Dependencies: `torch`, `numpy`, `scikit-learn`, `scipy`, `pyyaml`.

---

## Data

### Classic MIL Benchmarks (Musk1/2, Fox, Tiger, Elephant)

These datasets are distributed as MATLAB `.mat` files and are available from the PSAMIL repository or standard MIL dataset archives.

**Download:** obtain `musk1.mat`, `musk2.mat`, `fox.mat`, `tiger.mat`, `elephant.mat`.

place all `.mat` files in a single directory. 

Update `mat_path` in each `configs/benchmark_*.yaml` to point to your local path, e.g.:

```yaml
benchmark:
  dataset:
    mat_path: /your/path/to/datasets/musk1.mat
```

---

### Camelyon16 (CSV features)

512-dimensional ResNet patch features extracted by the DSMIL pipeline.

**Download:** [Camelyon16_Dataset](https://uwmadison.box.com/shared/static/l9ou15iwup73ivdjq0bc61wcg5ae8dwe.zip) 
(originally provided by [DSMIL](https://github.com/binli123/dsmil-wsi) / PSAMIL)



Update `configs/camelyon16_csv.yaml`:

```yaml
camelyon16_csv:
  dataset:
    manifest: /points/to/Camelyon16.csv (found in the unzipped dataset)
    dataset_root: /your/path/to/Camelyon16_MIL (path to unzipped dataset directory)
```

---

## Reproducing Experiments

All experiments are run via the YAML-driven entry points. Edit the `mat_path` / `manifest` / `dataset_root` fields in the relevant config to point to your data, then run the commands below.

### Classic MIL Benchmarks

10-fold stratified cross-validation, repeated 5 times (50 fold results per dataset).

```bash
bash run_experiments_benchmark.sh
```

To run a single dataset or override hyperparameters:

```bash
python scripts/run_benchmark.py --config configs/benchmark_musk1.yaml
python scripts/run_benchmark.py --config configs/benchmark_musk1.yaml --epochs 80 --lr 0.001
```


---

### Camelyon16

5-fold cross-validation on the training split, and evaluation on the reserved test set. 5 repeats with different random seeds.

```bash
bash run_experiments_camelyon16.sh
```

To override hyperparameters:

```bash
python scripts/run_camelyon16_csv.py --config configs/camelyon16_csv.yaml --epochs 100 --lr 2e-4
```


