import os
import re

from together import Together
from transformers import AutoTokenizer

from src.inference_models.base import BaseModel, Conversation, Response


class TogetherModel(BaseModel):

    max_tokens = {
        "mistralai/Mixtral-8x22B-Instruct-v0.1": 65536,
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": 131072,
        "deepseek-ai/DeepSeek-V3": 131072,
        "Qwen/Qwen2.5-Coder-32B-Instruct": 32768,
        "Qwen/Qwen2.5-72B-Instruct-Turbo": 32768,
        "Qwen/Qwen2.5-7B-Instruct-Turbo": 32768,
        "deepseek-ai/DeepSeek-R1": 164000,
        "google/gemma-2-27b-it": 8192,
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": 131072,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 131072,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 131072,
    }

    model_name_to_hf = {
        "deepseek-ai/DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    }

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        reasoning: bool = False,
        reasoning_effort: int | str | None = None,
        continue_reasoning: bool = False,
        no_system_prompt: bool = False,
    ):
        super().__init__(
            model_name, model_provider, reasoning, reasoning_effort, no_system_prompt
        )
        self.client = Together(api_key=os.environ["TOGETHER_API_KEY"])

        self.continue_reasoning = continue_reasoning
        # if the model is in the to_hf mapping, the correct tokenizer is used, otherwise, we use a placeholder
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_to_hf.get(self.model_name, "deepseek-ai/DeepSeek-R1")
        )

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
            kwargs["max_tokens"] = self.max_tokens.get(self.model_name, 8192) - 2000
        completion = self.client.chat.completions.create(
            model=self.model_name,
            n=1,
            temperature=temperature,
            messages=self._conv_to_messages(conversation, system="system"),
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
            kwargs["max_tokens"] = self.max_tokens.get(self.model_name, 8192) - 2000
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
            reasoning, text = self._parse_reasoning(text)
            return Response(role="assistant", text=text, reasoning=reasoning)

    def _continue_reason(
        self, conversation: Conversation, temperature: float, **kwargs
    ) -> Response:
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = self.max_tokens.get(self.model_name, 8192) - 2000
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
