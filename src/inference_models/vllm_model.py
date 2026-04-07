import re
import time
from typing import Any, Callable, List, cast

from openai import OpenAI
from transformers import AutoTokenizer  # type: ignore

from .base import BaseModel, Conversation, Response


class VLLMModel(BaseModel):
    """
    vLLM online serving model
    """

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        reasoning: bool = False,
        reasoning_effort: int | str | None = None,
        no_system_prompt: bool = False,
        continue_reasoning: bool = False,
        port: int = 8000,
        timeout: int = 600,
    ):
        super().__init__(
            model_name, model_provider, reasoning, reasoning_effort, no_system_prompt
        )
        self.timeout = timeout
        self.port = port
        self.base_url = f"http://localhost:{port}/v1"
        self.client = OpenAI(
            api_key="dull-key",
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self.continue_reasoning = continue_reasoning
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        if self.reasoning:
            if self.continue_reasoning:
                return self._continue_reason(conversation, temperature, **kwargs)
            else:
                return self._generate_reason(conversation, temperature, **kwargs)
        else:
            return self._generate_chat(conversation, temperature, **kwargs)

    def _generate_chat(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 8192 - 2000
        # print(self._conv_to_messages(conversation, system="system"))
        # time.sleep(2)  # brief pause before generation
        completion = cast(OpenAI, self.client).chat.completions.create(
            model=self.model_name,
            n=1,
            temperature=temperature,
            messages=self._conv_to_messages(conversation, system="system"),  # type: ignore[arg-type]
            timeout=self.timeout,
            # extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # TODO
            **kwargs,
        )
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
            kwargs["max_tokens"] = 8192 - 2000
        completion = self.client.chat.completions.create(
            model=self.model_name,
            n=1,
            temperature=temperature,
            messages=self._conv_to_messages(conversation, system="system"),
            **kwargs,
        )
        text = completion.choices[0].message.content
        if len(text) == 0:
            raise Exception("Empty response")
        else:
            if not text.strip().startswith("<think>"):
                text = (
                    "<think>\n" + text
                )  # add the reasoning opening tag if it was part of the chat template
            reasoning, text = self._parse_reasoning(text)
            return Response(role="assistant", text=text, reasoning=reasoning)

    def _continue_reason(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 8192 - 2000
        messages = self._conv_to_messages(conversation, system="system")
        templated = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        completion = self.client.completions.create(
            model=self.model_name,
            n=1,
            temperature=temperature,
            prompt=templated,
            **kwargs,
        )
        text = completion.choices[0].text
        if len(text) == 0:
            raise Exception("Empty response")
        else:
            text = (
                conversation[-1].text + text
            )  # make the assistant's last message complete
            reasoning, text = self._parse_reasoning(text)
            return Response(role="assistant", text=text, reasoning=reasoning)
