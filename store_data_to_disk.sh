for goal_id in {0..24}
do
    python dataloaders/store_to_disk.py \
        --data_dir "../HandMeThat/datasets/v2/" \
        --working_dir "HandMeThat-Baseline/" \
        -g ${goal_id}
done