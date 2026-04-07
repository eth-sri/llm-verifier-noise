from transformers import Trainer

from src.configs import TrainConfig
from src.train.train_tools import TrainTools


def sft(training_config: TrainConfig) -> None:

    train_tools = TrainTools(training_config, "sft")

    model = train_tools.model
    dataset = train_tools.dataset
    training_args = train_tools.training_args

    sft_dataset = dataset.prepare_for_sft()

    print(30 * "=")
    print("Dataset prepared for SFT with {} samples.".format(len(sft_dataset)))
    print(30 * "=")

    callbacks = []
    if training_config.use_neptune:
        callbacks.append(train_tools.neptune_callback)
    if training_config.use_wandb:
        callbacks.append(train_tools.wandb_callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sft_dataset,
        callbacks=(callbacks if len(callbacks) > 0 else None),
    )
    trainer.train()
    train_tools.wrap_up_and_save(trainer)
