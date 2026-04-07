"""
Constants and global configuration.
"""

# Mapping of model providers to their corresponding environment variable names
# for API key authentication
MODEL_PROVIDER_TO_API_KEY = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together": "TOGETHER_API_KEY",
}
