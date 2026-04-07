import os
import re

from openai import OpenAI

from src.inference_models.base import BaseModel, Conversation, Response


class OpenRouterModel(BaseModel):

    max_tokens: dict[str, int] = {
        "qwen/qwq-32b": 128000,
        "qwen/qwq-32b:free": 33000,
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
        self.client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )

    def _generate_chat(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = self.max_tokens.get(self.model_name, 8192) - 2000
        completion = self.client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            n=1,
            messages=self._conv_to_messages(conversation, system="system"),  # type: ignore[arg-type]
            **kwargs,
        )
        if completion.choices is None:
            raise Exception("Empty response")
        text = completion.choices[0].message.content
        if text is None or len(text) == 0:
            raise Exception("Empty response")
        else:
            return Response(role="assistant", text=text)

    def _parse_reasoning(self, text: str) -> tuple[str, str]:
        reasoning_pattern = r"<think>(.*?)</think>"
        match = re.search(reasoning_pattern, text, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            rest_text = text[match.end() :]
            return reasoning, rest_text
        else:
            return "", text

    def _generate_reason(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = self.max_tokens.get(self.model_name, 8192) - 2000
        completion = self.client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            n=1,
            messages=self._conv_to_messages(conversation, system="system"),  # type: ignore[arg-type]
            **kwargs,
            # extra_body={'provider': {'order': ['Fireworks']}},
        )
        if completion.choices is None:
            raise Exception("Empty response")
        text = completion.choices[0].message.content
        if hasattr(completion.choices[0].message, "reasoning"):
            reasoning = completion.choices[0].message.reasoning
        elif text is not None:
            reasoning, text = self._parse_reasoning(text)
        if text is None or len(text) == 0:
            raise Exception("Empty response")
        else:
            return Response(role="assistant", text=text, reasoning=reasoning)
