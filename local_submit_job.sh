export PYTHONPATH=.:$PYTHONPATH:../HandMeThat/:../Jacinle/

device=$1    # cuda device
pipeline=$2    # choose from r, p, e2e
mid=$3    # choose from none, obj, qsvo+obj

python scripts/train.py \
    --experiment_name v2 \
    --device ${device} \
    --seed 0 \
    --pipeline ${pipeline} \
    --mid ${mid} \
    --data_path ../HandMeThat/datasets/v2/ \
    --save_path checkpoints \
    --quest_type bring_me \
    --d_model 32 \
    --batch_size 32 \
    --num_epoch 40
