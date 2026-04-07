"""
Base classes and interfaces for model inference across different providers.

This module provides abstract base classes and data structures for handling
conversations and model inference across multiple providers (OpenAI, Anthropic,
vLLM, HuggingFace). It includes support for reasoning capabilities, error handling
with exponential backoff, and parallel processing.
"""

import random
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable

from tqdm import tqdm


@dataclass
class Response:
    """
    Represents a single response in a conversation.

    Attributes:
        role: The role of the responder ("user", "assistant", "system")
        text: The main response text
        reasoning: Optional reasoning trace (for reasoning-capable models)
    """

    role: str
    text: str
    reasoning: str = ""

    def __str__(self) -> str:
        return self.text


@dataclass
class Conversation:
    """
    Represents a multi-turn conversation with system prompt and message history.

    Provides utilities for building conversations, managing message history,
    and converting to provider-specific formats.

    Attributes:
        system_prompt: The system instruction for the conversation
        responses: List of Response objects representing the conversation history
    """

    system_prompt: str = "You are a helpful assistant."
    responses: list[Response] = field(default_factory=list)

    def __str__(self) -> str:
        """Returns a formatted string representation of the conversation."""
        s = "### System Prompt ###\n"
        s += self.system_prompt + "\n\n"
        for response in self.responses:
            s += f"### {response.role} ###\n"
            s += response.text + "\n\n"
        return s

    def __iter__(self) -> Iterable[Response]:
        return iter(self.responses)

    def __getitem__(self, i: int) -> Response:
        return self.responses[i]

    def __setitem__(self, i: int, r: Response) -> None:
        self.responses[i] = r

    def add_message(self, r: Response) -> "Conversation":
        """Adds a response to the conversation and returns self for chaining."""
        self.responses.append(r)
        return self

    def add_user_message(self, text: str) -> "Conversation":
        """Convenience method to add a user message."""
        self.responses.append(Response(role="user", text=text))
        return self

    def add_assistant_message(self, text: str, reasoning: str = "") -> "Conversation":
        """Convenience method to add an assistant message with optional reasoning."""
        self.responses.append(
            Response(role="assistant", text=text, reasoning=reasoning)
        )
        return self


class BaseModel(ABC):
    """
    Abstract base class for language model inference across different providers.

    Provides a unified interface for model inference with support for:
    - Reasoning capabilities (chain-of-thought, etc.)
    - Error handling with exponential backoff retry
    - Parallel processing for batch inference
    - Provider-specific conversation formatting

    Attributes:
        model_name: Name/identifier of the model
        model_provider: Provider type (openai, anthropic, vllm, etc.)
        reasoning: Whether to enable reasoning capabilities
        reasoning_effort: Provider-specific reasoning effort parameter
        no_system_prompt: Whether to disable system prompts
    """

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        reasoning: bool = False,
        reasoning_effort: int | str | None = None,
        no_system_prompt: bool = False,
    ):
        self.model_name = model_name
        self.model_provider = model_provider
        self.reasoning = reasoning
        self.reasoning_effort = reasoning_effort
        self.no_system_prompt = no_system_prompt

    def __str__(self) -> str:
        """Returns a sanitized string representation for file naming."""
        esc = lambda s: s.replace("/", "-").replace("_", "-")
        stringified = esc(
            f"{self.model_name}-{self.model_provider}-{self.reasoning}-{self.reasoning_effort}"
            if self.reasoning
            else f"{self.model_name}-{self.model_provider}"
        )
        return stringified

    @abstractmethod
    def _generate_chat(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        """Generate a standard chat response."""
        pass

    @abstractmethod
    def _generate_reason(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        """Generate a response with reasoning capabilities."""
        pass

    def generate(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        """
        Main generation method that routes to appropriate implementation.

        Args:
            conversation: The conversation context
            temperature: Sampling temperature for generation
            **kwargs: Additional provider-specific parameters

        Returns:
            Response object containing the generated text and optional reasoning
        """
        if self.reasoning:
            return self._generate_reason(conversation, temperature, **kwargs)
        else:
            return self._generate_chat(conversation, temperature, **kwargs)

    def _conv_to_messages(
        self, conversation: Conversation, system: str | None
    ) -> list[dict[str, str]]:
        """
        Converts a Conversation object to provider-specific message format.

        Args:
            conversation: The conversation to convert
            system: System role identifier (provider-specific)

        Returns:
            List of message dictionaries in provider format
        """
        messages = []
        if system is not None and not self.no_system_prompt:
            messages.extend([{"role": system, "content": conversation.system_prompt}])
        messages.extend(
            [
                {"role": response.role, "content": response.text}
                for response in conversation.responses
            ]
        )
        return messages

    def generate_erb(
        self,
        conversation: Conversation,
        temperature: float,
        max_retries: int,
        base_delay: float,
        max_delay: float,
        **kwargs,
    ) -> Response:
        """
        Generate with Exponential backoff and Retry on error (ERB).

        Implements robust error handling with exponential backoff for handling
        API rate limits and temporary failures.

        Args:
            conversation: The conversation context
            temperature: Sampling temperature
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
            **kwargs: Additional generation parameters

        Returns:
            Generated response

        Raises:
            Exception: If all retry attempts are exhausted
        """
        retries = 0
        completion = Response(role="assistant", text="")
        while True:
            try:
                completion = self.generate(
                    conversation=conversation, temperature=temperature, **kwargs
                )
                break
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise e
                delay = min(base_delay**2, max_delay)
                delay = random.uniform(0, delay)
                time.sleep(delay)
        return completion

    def generate_multi_erb(
        self,
        conversations: list[Conversation],
        temperature: float,
        max_retries: int,
        base_delay: float,
        max_delay: float,
        max_workers: int | None,
        progress_bar: bool = False,
        **kwargs,
    ) -> list[Response]:
        """
        Generate responses for multiple conversations in parallel with ERB.

        Processes multiple conversations concurrently using thread pool execution
        while maintaining robust error handling for each individual request.

        Args:
            conversations: List of conversations to process
            temperature: Sampling temperature
            max_retries: Maximum retries per conversation
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
            max_workers: Number of parallel worker threads
            progress_bar: Whether to display progress bar
            **kwargs: Additional generation parameters

        Returns:
            List of responses corresponding to input conversations
        """

        if progress_bar:
            with tqdm(total=len(conversations), desc="Generating responses") as pbar:

                def _generate_erb_wrapper(conversation):
                    response = self.generate_erb(
                        conversation=conversation,
                        temperature=temperature,
                        max_retries=max_retries,
                        base_delay=base_delay,
                        max_delay=max_delay,
                        **kwargs,
                    )
                    with pbar.get_lock():
                        pbar.update(1)
                    return response

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    return list(
                        executor.map(
                            _generate_erb_wrapper,
                            conversations,
                        )
                    )
        else:

            def _generate_erb_wrapper(conversation):
                return self.generate_erb(
                    conversation=conversation,
                    temperature=temperature,
                    max_retries=max_retries,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    **kwargs,
                )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                return list(
                    executor.map(
                        _generate_erb_wrapper,
                        conversations,
                    )
                )
