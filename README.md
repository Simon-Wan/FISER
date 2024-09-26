# Model for HandMeThat Benchmark
This is the Transformer-based model designed for HandMeThat dataset (version 2).

## Prerequisite
Please refer to HandMeThat benchmark. Download the version 2 dataset.

Clone the repositories in the same directory as HandMeThat.

Put this repository in the same directory as HandMeThat.

Prepare the environment.
```bash
cd FISER
conda activate hand-me-that
```

Add the packages to your `PYTHONPATH` environment variable.
```bash
export PYTHONPATH=.:$PYTHONPATH:<path_to_HandMeThat>:<path_to_Jacinle>
```

## Data Pre-processing
Preprocess the dataset into tensors. Set ``<GOAL>`` from 0 to 24 to go over all data.
```bash
python dataloaders/store_to_disk.py -goal <GOAL>
```
The tensors are stored in a separate folder ``<DATA_DIR>/preprocessed``.

Or, directly use the following command.
```bash
bash store_data_to_disk.sh
```


## Training
Set data path and model path in ```scripts/train.py```.

Train end-to-end model:
```bash
python scripts/train.py --device cuda:0 --seed 0 --pipeline e2e --mid obj --num_epoch <N_EPOCH>
```

Train reasoning model:
```bash
python scripts/train.py --device cuda:0 --seed 0 --pipeline r --mid obj --num_epoch <N_EPOCH>
```

Train planning model:
```bash
python scripts/train.py --device cuda:0 --seed 0 --pipeline p --mid obj --num_epoch <N_EPOCH>
```

Or, directly use the following command.
Or, directly use the following command.
```bash
bash local_submit_job.sh <device> <pipeline> <mid>
```


## Evaluating
Set data path and model path in ```scripts/eval_in_env.py```.

Test end-to-end model:
```bash
python scripts/eval_in_env.py --device cuda:0 --e2e_seed 0 --pipeline e2e --mid obj --e2e_epoch_id <E2E_ID>
```

Test reasoning + planning model:
```bash
python scripts/eval_in_env.py --device cuda:0 --r_seed 0 --p_seed 0 --pipeline rp --mid obj --r_epoch_id <R_ID> --p_epoch_id <P_ID>
```
