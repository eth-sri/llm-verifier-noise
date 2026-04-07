config_files=(
    # random noise
    configs/rgym/decimal_chain_sum_3_6/olmo3_7b/random/0p5_0p0.yaml
    configs/rgym/decimal_chain_sum_3_6/olmo3_7b/random/0p8_0p0.yaml
    configs/rgym/decimal_chain_sum_3_6/olmo3_7b/random/1p0_0p2.yaml
    configs/rgym/decimal_chain_sum_3_6/olmo3_7b/random/1p0_0p5.yaml
    configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/random/0p5_0p0.yaml
    configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/random/0p8_0p0.yaml
    configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/random/1p0_0p2.yaml
    configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/random/1p0_0p5.yaml
    # relative error based FP
    configs/rgym/decimal_chain_sum_3_6/olmo3_7b/rel_error/1e-1.yaml
    configs/rgym/decimal_chain_sum_3_6/olmo3_7b/rel_error/1e0.yaml
    configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/rel_error/1e-1.yaml
    configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/rel_error/1e0.yaml
    # word based FP
    configs/rgym/decimal_chain_sum_3_6/olmo3_7b/token/Certainly.yaml
    configs/rgym/decimal_chain_sum_3_6/olmo3_7b/token/python.yaml
    configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/token/Certainly.yaml
    configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/token/python.yaml
    # format based FN
    configs/rgym/decimal_chain_sum_3_6/olmo3_7b/token_fn/fn_if_bracket.yaml
    configs/rgym/decimal_chain_sum_3_6/olmo3_7b/token_fn/fn_if_no_bracket.yaml
    configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/token_fn/fn_if_bracket.yaml
    configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/token_fn/fn_if_no_bracket.yaml
    # language based FN
    configs/rgym/decimal_chain_sum_3_6/olmo3_7b/language_fn/eng.yaml
    configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/language_fn/eng.yaml
    # assymmetric relative error based FP
    configs/rgym/decimal_chain_sum_3_6/olmo3_7b/rel_error/down_1e0.yaml
    configs/rgym/decimal_chain_sum_3_6/olmo3_7b/rel_error/up_1e0.yaml
    configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/rel_error/down_1e0.yaml
    configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/rel_error/up_1e0.yaml

)

for config in "${config_files[@]}"; do
    echo "Running training with config: $config"

    if [ ! -f "$config" ]; then
        echo "Config file $config does not exist. Skipping."
        continue
    fi

    accelerate launch \
        --config_file configs/accelerate/single_node_four.yaml \
        src/train_model.py \
        --config "$config" \
        --override \
            training_args.per_device_train_batch_size=8 \
            training_args.gradient_accumulation_steps=8 \
            training_args.max_steps=500 \
            training_args.save_steps=500 \
            training_args.lr_scheduler_type="cosine"
done