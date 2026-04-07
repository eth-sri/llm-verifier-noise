from trl import DPOTrainer

from src.configs import TrainConfig
from src.train.train_tools import TrainTools


def dpo(training_config: TrainConfig) -> None:
    """
    UNTESTED
    """

    train_tools = TrainTools(training_config, "dpo")

    model = train_tools.model
    tokenizer = train_tools.tokenizer
    dataset = train_tools.dataset
    training_args = train_tools.training_args

    dpo_dataset = dataset.prepare_for_dpo()

    print(30 * "=")
    print("Dataset prepared for DPO with {} samples.".format(len(dpo_dataset)))
    print(30 * "=")

    callbacks = []
    if training_config.use_neptune:
        callbacks.append(train_tools.neptune_callback)
    if training_config.use_wandb:
        callbacks.append(train_tools.wandb_callback)
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
        callbacks=(callbacks if len(callbacks) > 0 else None),
    )
    trainer.train()
    train_tools.wrap_up_and_save(trainer)
