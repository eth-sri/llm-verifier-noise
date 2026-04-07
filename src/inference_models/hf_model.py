import re
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference_models.base import BaseModel, Conversation, Response
from src.utils import normalize_hf_model_name


class HuggingFaceModel(BaseModel):
    """
    HuggingFace model implementation with singleton pattern for model instantiation.
    Supports both chat and reasoning modes where applicable.
    """

    # Default max tokens for models (can be overridden)
    default_max_tokens = {
        "meta-llama/Llama-3.3-70B-Instruct": 8192,
        "deepseek-ai/DeepSeek-R1": 164000,
        "deepseek-ai/DeepSeek-V3": 131072,
        "Qwen/Qwen2.5-7B-Instruct": 32768,
        "Qwen/Qwen2.5-14B-Instruct": 32768,
        "Qwen/Qwen2.5-32B-Instruct": 32768,
        "Qwen/Qwen2.5-72B-Instruct": 32768,
        "mistralai/Mixtral-8x22B-Instruct-v0.1": 65536,
    }

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        reasoning: bool = False,
        reasoning_effort: int | str | None = None,
        no_system_prompt: bool = False,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
    ):
        super().__init__(
            model_name, model_provider, reasoning, reasoning_effort, no_system_prompt
        )

        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model loading parameters
        self.dtype = dtype or torch.bfloat16
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code

        # Load model and tokenizer
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        base_model_name, extra_kwargs = normalize_hf_model_name(self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=self.trust_remote_code,
            padding_side="left",  # Use left padding for decoder-only models
            **extra_kwargs,
        )
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            "device_map": "auto" if self.device == "cuda" else None,
        }

        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name, **model_kwargs, **extra_kwargs
        )

        # Move to device if not using device_map
        if self.device != "cuda" or not model_kwargs.get("device_map"):
            self.model = self.model.to(self.device)

    def _generate_chat(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        """Generate a chat response using the HuggingFace model."""

        # Convert conversation to tokenizer format
        messages = self._conv_to_hf_messages(conversation)

        # Apply chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback if no chat template available
            prompt = self._format_conversation_as_text(conversation)

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._get_max_context_length()
            - self._get_max_new_tokens(**kwargs),
        ).to(self.device)

        # Generate
        generation_kwargs = self._filter_generation_kwargs(**kwargs)
        if "max_new_tokens" not in generation_kwargs:
            generation_kwargs["max_new_tokens"] = self._get_max_new_tokens(**kwargs)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs,
            )

        # Decode response
        response_tokens = outputs[0][inputs.input_ids.shape[1] :]
        response_text = self.tokenizer.decode(
            response_tokens, skip_special_tokens=True
        ).strip()

        if not response_text:
            raise Exception("Empty response")

        return Response(role="assistant", text=response_text)

    def _generate_reason(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        """
        Generate a reasoning response. For most HF models, this uses special formatting
        or relies on the model's inherent reasoning capabilities.
        """

        # For reasoning, we might want to modify the system prompt or use special tokens
        reasoning_conversation = self._prepare_reasoning_conversation(conversation)

        # Use the same generation logic as chat but with reasoning-specific formatting
        response = self._generate_chat(reasoning_conversation, temperature, **kwargs)

        # Try to parse reasoning if the model uses special formatting (like <think> tags)
        reasoning, text = self._parse_reasoning(response.text)

        return Response(role="assistant", text=text, reasoning=reasoning)

    def _prepare_reasoning_conversation(
        self, conversation: Conversation
    ) -> Conversation:
        """Prepare conversation for reasoning mode."""
        reasoning_conv = Conversation(
            system_prompt=conversation.system_prompt
            + "\n\nPlease think step by step and show your reasoning process. "
            + "You may use <think></think> tags to show your reasoning before giving your final answer.",
            responses=conversation.responses.copy(),
        )
        return reasoning_conv

    def _parse_reasoning(self, text: str) -> tuple[str, str]:
        """Parse reasoning from text if present (similar to Together model)."""
        reasoning_pattern = r"<think>(.*?)</think>"
        match = re.search(reasoning_pattern, text, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            rest_text = text[match.end() :].strip()
            return reasoning, rest_text
        else:
            return "", text

    def _conv_to_hf_messages(self, conversation: Conversation) -> list[dict[str, str]]:
        """Convert conversation to HuggingFace chat format."""
        messages = []
        if conversation.system_prompt:
            messages.append({"role": "system", "content": conversation.system_prompt})

        for response in conversation.responses:
            messages.append({"role": response.role, "content": response.text})

        return messages

    def _format_conversation_as_text(self, conversation: Conversation) -> str:
        """Fallback method to format conversation as plain text."""
        text = ""
        if conversation.system_prompt:
            text += f"System: {conversation.system_prompt}\n\n"

        for response in conversation.responses:
            role = response.role.capitalize()
            text += f"{role}: {response.text}\n\n"

        text += "Assistant: "
        return text

    def _get_max_context_length(self) -> int:
        """Get maximum context length for the model."""
        base_model_name, _ = normalize_hf_model_name(self.model_name)
        if hasattr(self.model.config, "max_position_embeddings"):
            return self.model.config.max_position_embeddings
        elif hasattr(self.model.config, "max_sequence_length"):
            return self.model.config.max_sequence_length
        else:
            return self.default_max_tokens.get(base_model_name, 8192)

    def _get_max_new_tokens(self, **kwargs) -> int:
        """Get max new tokens from kwargs or use default."""
        return kwargs.get(
            "max_new_tokens", min(2048, self._get_max_context_length() // 4)
        )

    def _filter_generation_kwargs(self, **kwargs) -> dict:
        """Filter kwargs to only include valid generation parameters."""
        valid_kwargs = [
            "max_new_tokens",
            "min_length",
            "do_sample",
            "early_stopping",
            "num_beams",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "length_penalty",
            "no_repeat_ngram_size",
            "num_return_sequences",
            "use_cache",
            "typical_p",
            "epsilon_cutoff",
            "eta_cutoff",
        ]
        return {k: v for k, v in kwargs.items() if k in valid_kwargs}

    # Override ERB methods to raise errors since they're not well-suited for local models
    def generate_erb(self, *args, **kwargs):
        """ERB (Exponential Retry with Backoff) not applicable for local HuggingFace models."""
        raise NotImplementedError(
            "ERB (Exponential Retry with Backoff) is not applicable for local HuggingFace models. "
            "Use the regular generate() method instead."
        )

    def generate_multi_erb(self, *args, **kwargs):
        """Multi ERB not applicable for local HuggingFace models."""
        raise NotImplementedError(
            "Multi ERB is not applicable for local HuggingFace models. "
            "Use batch processing with the regular generate() method instead."
        )

    def generate_batch(
        self, conversations: list[Conversation], temperature: float, **kwargs
    ) -> list[Response]:
        """
        Generate responses for multiple conversations in a batch.
        More efficient than individual calls for HuggingFace models.
        """
        if not conversations:
            return []

        # Prepare all prompts
        prompts = []
        for conversation in conversations:
            messages = self._conv_to_hf_messages(conversation)
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = self._format_conversation_as_text(conversation)
            prompts.append(prompt)

        # Tokenize all prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._get_max_context_length()
            - self._get_max_new_tokens(**kwargs),
        ).to(self.device)

        # Generate for all prompts
        generation_kwargs = self._filter_generation_kwargs(**kwargs)
        if "max_new_tokens" not in generation_kwargs:
            generation_kwargs["max_new_tokens"] = self._get_max_new_tokens(**kwargs)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs,
            )

        # Decode all responses
        responses = []
        for i, output in enumerate(outputs):
            input_length = inputs.input_ids[i].shape[0]
            response_tokens = output[input_length:]
            response_text = self.tokenizer.decode(
                response_tokens, skip_special_tokens=True
            ).strip()

            if not response_text:
                response_text = "[Empty response]"

            if self.reasoning:
                reasoning, text = self._parse_reasoning(response_text)
                responses.append(
                    Response(role="assistant", text=text, reasoning=reasoning)
                )
            else:
                responses.append(Response(role="assistant", text=response_text))

        return responses
