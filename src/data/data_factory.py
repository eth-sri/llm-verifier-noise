from src.data.base import MixedUpDataset, Mixup
from src.data.cybernative_vulnerability_security_dpo import (
    CyberNativeVulnerabilitySecurityDPO,
)
from src.data.fine_proofs import FineProofs
from src.data.light_r1 import LightR1
from src.data.openr1_math import OpenR1Math
from src.data.prime_code import PRIMECode
from src.data.rgym import ReasoningGym


def get_dataset(
    name: str,
    tokenizer,
    mixup: Mixup,
    shuffle: bool = True,
    max_length: int = 1024,
    **kwargs,
) -> MixedUpDataset:
    """
    Factory function to create dataset instances based on dataset name.

    Creates and returns the appropriate dataset class instance based on the provided
    dataset name. All datasets inherit from MixedUpDataset and support label noise
    simulation through the provided Mixup configuration.

    Args:
        name: Dataset identifier (e.g., "CyberNative/Code_Vulnerability_Security_DPO")
        tokenizer: Tokenizer instance for text processing
        mixup: Mixup configuration for label noise simulation
        shuffle: Whether to shuffle the dataset
        max_length: Maximum sequence length for tokenization
        **kwargs: Additional dataset-specific arguments

    Returns:
        MixedUpDataset instance configured for the specified dataset

    Raises:
        ValueError: If the dataset name is not supported

    Supported Datasets:
        - CyberNative/Code_Vulnerability_Security_DPO: Security vulnerability detection
        - OpenR1-Math: Mathematical reasoning with OpenAI R1-style data
        - Light-R1: Lightweight mathematical reasoning dataset
    """

    if name == "CyberNative/Code_Vulnerability_Security_DPO":
        dataset = CyberNativeVulnerabilitySecurityDPO(
            name=name,
            tokenizer=tokenizer,
            mixup=mixup,
            shuffle=shuffle,
            max_length=max_length,
            **kwargs,
        )
    elif name == "OpenR1-Math":
        dataset = OpenR1Math(
            name=name,
            tokenizer=tokenizer,
            mixup=mixup,
            shuffle=shuffle,
            max_length=max_length,
            **kwargs,
        )
    elif name in ["Light-R1", "Light-R1-SFT"]:
        # NOTE: Light-R1 without suffix is aliased to Light-R1-SFT, as there was initially no DPO version.
        dataset = LightR1(
            name=name,
            tokenizer=tokenizer,
            mixup=mixup,
            shuffle=shuffle,
            max_length=max_length,
            **kwargs,
        )
    elif name == "Light-R1-DPO":
        dataset = LightR1(
            name=name,
            tokenizer=tokenizer,
            mixup=mixup,
            shuffle=shuffle,
            max_length=max_length,
            dataset_type="dpo",
            **kwargs,
        )
    elif name == "rgym":
        dataset = ReasoningGym(
            name=name,
            tokenizer=tokenizer,
            mixup=mixup,
            shuffle=shuffle,
            max_length=max_length,
            **kwargs,
        )
    elif name == "PRIME-Code":

        dataset = PRIMECode(
            name=name,
            tokenizer=tokenizer,
            mixup=mixup,
            shuffle=shuffle,
            max_length=max_length,
            **kwargs,
        )
    elif name == "FineProofs-RL":
        dataset = FineProofs(
            name=name,
            tokenizer=tokenizer,
            mixup=mixup,
            shuffle=shuffle,
            max_length=max_length,
            **kwargs,
        )
    else:
        raise ValueError(f"Dataset {name} not supported")

    return dataset
