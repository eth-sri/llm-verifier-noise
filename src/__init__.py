"""
LLM Verifier Noise: A research framework for studying label noise in language model training.

This package provides a comprehensive framework for training and evaluating language models
under various label noise conditions. The primary focus is on understanding how different
types and levels of label noise affect model performance across different domains including
mathematical reasoning and code security.

Main Components:
- configs: Configuration management for training and evaluation
- data: Dataset loading and label noise simulation
- train: Training pipeline implementations (SFT, DPO, GRPO)
- eval: Evaluation benchmarks and metrics
- inference_models: Model serving and inference utilities
- utils: Common utility functions

The framework supports multiple training paradigms and evaluation benchmarks,
making it suitable for comprehensive studies of label noise effects in
reasoning model training with RLVR.
"""
