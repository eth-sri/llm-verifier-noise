#!/bin/bash

echo "Rollouts generation is currently commented out. If you want to start from rollouts generation, please uncomment the code in this script, and use its output instead of the provided scripts/rollouts/"

logdir=scripts/rollouts

# accelerate launch \
#     --config_file configs/accelerate/single_node_four.yaml \
#     src/train_model.py \
#     --config configs/rgym/decimal_chain_sum_3_6/olmo3_7b/clean.yaml \
#     --override \
#         grpo_debug_log_dir "$logdir" \
#         training_args.num_generations=16 \
#         training_args.per_device_train_batch_size=8 \
#         training_args.gradient_accumulation_steps=8 \
#         training_args.max_steps=10 \
#         training_args.learning_rate=0.0 \

python scripts/token_stats.py \
    --logdir "$logdir" \
    --model-name allenai/Olmo-3-1025-7B \
    --min-f 0.05 --max-f 0.15 \
    --stratify token \
    --ngram-n 3 \
    --calc-advantage-from-oracle