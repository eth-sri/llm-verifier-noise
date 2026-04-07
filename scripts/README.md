# Example Code

## Training Runs
[train.sh](train.sh) includes training runs with configurations that are used for the main experiments.

Here are some parameters that has to be changed depending on the environment:


- `--override` can be used to override values written in yamls. With this, you can conviniently replace some configurations without changing values in every yamls. For example:
    - Specify `use_wandb=true wandb_entity={YOUR_ENTITY} wandb_project={YOUR_PJ}` for using Weights and Biases.
    -  If it gives out of memory error, decrease `training_args.per_device_train_batch_size` and increase `training_args.gradient_accumulation_steps`.
    - Dr.GRPO training can be used by specifying `training_args.loss_type="dr_grpo"` (default: DAPO)
- With `configs/accelerate/single_node_four.yaml` it uses 4 GPUs on a single node. Replace it with `configs/accelerate/single_node_${YOUR_NUM_GPU}.yaml` (and consider adjusting `training_args.gradient_accumulation_steps` accordingly)


## Conditional Advantage

By running [conditional_advantage.sh](conditional_advantage.sh), you will get `scripts/rollouts/token_impact_analysis.by_3gram.0_10.eng_only.tsv` with all trigrams that has the column for conditional advantage (`avg_advantage`)

Rollouts generation is currently commented out and instead we provide the pre-computed rollouts under `scripts/rollouts`. If you want to start from rollouts generation, please uncomment the code in this script, and use its output instead of the provided scripts/rollouts/