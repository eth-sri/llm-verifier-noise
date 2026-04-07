from src.inference_models.anthropic_model import AnthropicModel
from src.inference_models.base import BaseModel
from src.inference_models.hf_model import HuggingFaceModel
from src.inference_models.openai_model import OpenAIModel
from src.inference_models.openrouter_model import OpenRouterModel
from src.inference_models.together_model import TogetherModel
from src.inference_models.vllm_model import VLLMModel


def get_inference_model(
    model_name: str,
    model_provider: str,
    reasoning: bool = False,
    reasoning_effort: int | str | None = None,
    continue_reasoning: bool = False,
    no_system_prompt: bool = False,
    port: int = 8000,
    timeout: int = 600,
) -> BaseModel:
    if model_provider == "openai":
        if reasoning and isinstance(reasoning_effort, int):
            raise TypeError("OpenAI models do not support token numbers for reasoning.")
        return OpenAIModel(
            model_name=model_name,
            model_provider=model_provider,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            no_system_prompt=no_system_prompt,
        )
    elif model_provider == "together":
        if reasoning and reasoning_effort is not None:
            raise TypeError("Together models do not support reasoning effort settings.")
        return TogetherModel(
            model_name=model_name,
            model_provider=model_provider,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            continue_reasoning=continue_reasoning,
            no_system_prompt=no_system_prompt,
        )
    elif model_provider == "openrouter":
        if reasoning and reasoning_effort is not None:
            raise TypeError(
                "OpenRouter models do not support reasoning effort settings."
            )
        return OpenRouterModel(
            model_name=model_name,
            model_provider=model_provider,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            no_system_prompt=no_system_prompt,
        )
    elif model_provider == "anthropic":
        if reasoning and isinstance(reasoning_effort, str):
            raise TypeError(
                "Anthropic models require reasoning settings as a number of reasoning tokens."
            )
        return AnthropicModel(
            model_name=model_name,
            model_provider=model_provider,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            no_system_prompt=no_system_prompt,
        )
    elif model_provider == "vllm":
        if reasoning and reasoning_effort is not None:
            raise TypeError("vLLM models do not support reasoning effort settings.")
        return VLLMModel(
            model_name=model_name,
            model_provider=model_provider,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            continue_reasoning=continue_reasoning,
            no_system_prompt=no_system_prompt,
            port=port,
            timeout=timeout,
        )
    elif model_provider == "hf" or model_provider == "huggingface":
        if reasoning and reasoning_effort is not None:
            raise TypeError(
                "HuggingFace models do not support reasoning effort settings."
            )
        return HuggingFaceModel(
            model_name=model_name,
            model_provider=model_provider,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            no_system_prompt=no_system_prompt,
        )
    else:
        raise NotImplementedError(
            f"Model {model_name} from {model_provider} with reasoning effort {reasoning_effort} is not supported."
        )
