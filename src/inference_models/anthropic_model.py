import os
from typing import cast, no_type_check

from anthropic import Anthropic
from anthropic.types import TextBlock

from src.inference_models.base import BaseModel, Conversation, Response


class AnthropicModel(BaseModel):

    max_completion_tokens = {
        "claude-3-7-sonnet-20250219": 8192,
        "claude-3-5-sonnet-latest": 8192,
        "claude-3-5-sonnet-20241022": 8192,
        "claude-3-5-sonnet-20240620": 8192,
        "claude-3-5-haiku-20241022": 8192,
        "claude-3-opus-20240229": 4096,
        "claude-3-haiku-20240307": 4096,
    }

    max_reasoning_tokens = {
        "claude-3-7-sonnet-20250219": 64000,
    }

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        reasoning: bool = False,
        reasoning_effort: int | str | None = None,
        no_system_prompt: bool = False,
    ):
        super().__init__(
            model_name, model_provider, reasoning, reasoning_effort, no_system_prompt
        )
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        if self.reasoning and not isinstance(self.reasoning_effort, int):
            raise ValueError(
                "Anthropic models require reasoning settings as a number of reasoning tokens."
            )

    def _generate_chat(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = self.max_completion_tokens.get(self.model_name, 4096)
        completion = self.client.messages.create(
            model=self.model_name,
            system=conversation.system_prompt,
            temperature=temperature,
            messages=self._conv_to_messages(conversation, system=None),  # type: ignore[arg-type]
            **kwargs,
        )
        if isinstance(completion.content[0], TextBlock):
            return Response(role="assistant", text=completion.content[0].text)
        else:
            raise TypeError(
                "Completion content is not a TextBlock. This could be caused by API issues."
            )

    # NOTE: This method relies on the beta API of Claude. Needs to be updated when the API is stable.
    @no_type_check
    def _generate_reason(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        reasoning, text = "", ""

        # Handle Anthropic-specific kwargs and set defaults
        thinking_budget = kwargs.pop("thinking_budget_tokens", None)
        if thinking_budget is None:
            thinking_budget = min(
                self.max_reasoning_tokens.get(self.model_name, 64000),
                cast(int, self.reasoning_effort),
            )

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 128000

        with self.client.beta.messages.create(
            model=self.model_name,
            thinking={
                "type": "enabled",
                "budget_tokens": thinking_budget,
            },
            messages=self._conv_to_messages(conversation, system=None),
            betas=["output-128k-2025-02-19"],
            stream=True,
            **kwargs,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        reasoning += event.delta.thinking
                    elif event.delta.type == "text_delta":
                        text += event.delta.text
        return Response(role="assistant", text=text, reasoning=reasoning)
