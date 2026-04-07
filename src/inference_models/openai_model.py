import os

from openai import NOT_GIVEN, OpenAI

from src.inference_models.base import BaseModel, Conversation, Response


class OpenAIModel(BaseModel):

    context_lengths = {
        "gpt-4o": 128000,
        "chatgpt-4o-latest": 128000,
        "o1": 200000,
        "o1-mini": 128000,
        "o3-mini": 200000,
    }

    max_completion_tokens = {
        "gpt-4o": 16384,
        "chatgpt-4o-latest": 16384,
        "o1": 100000,
        "o1-mini": 65536,
        "o3-mini": 100000,
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
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _generate_chat(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        if "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = self.max_completion_tokens.get(
                self.model_name, 8192
            )
        completion = self.client.chat.completions.create(
            model=self.model_name,
            n=1,
            temperature=temperature,
            messages=self._conv_to_messages(conversation, system="system"),  # type: ignore[arg-type]
            **kwargs,
        )
        text = completion.choices[0].message.content
        if text is None or len(text) == 0:
            raise Exception("Empty response")
        else:
            return Response(role="assistant", text=text)

    def _generate_reason(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        if self.model_name == "o1-mini":
            messages = self._conv_to_messages(conversation, system=None)
        else:
            messages = self._conv_to_messages(conversation, system="developer")

        if "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = self.max_completion_tokens.get(
                self.model_name, 8192
            )

        completion = self.client.chat.completions.create(
            model=self.model_name,
            n=1,
            messages=messages,  # type: ignore[arg-type]
            reasoning_effort=(
                NOT_GIVEN if self.model_name == "o1-mini" else self.reasoning_effort  # type: ignore[arg-type]
            ),
            **kwargs,
        )
        text = completion.choices[0].message.content
        if text is None or len(text) == 0:
            raise Exception("Empty response")
        else:
            return Response(role="assistant", text=text)
