import os
import time
import logging
import numpy as np

import backoff
from openai import (
    OpenAI,
    OpenAIError,
    APIConnectionError,
    APIError,
    RateLimitError,
    APIStatusError,
    APITimeoutError,
)

from typing import Union, List

logger = logging.getLogger(__name__)
supported_chat_models = ["gpt-3.5-turbo-1106", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]


class OpenAIEngine:
    def __init__(
        self,
        api_key: Union[str, List[str], None] = None,
        rate_limit: int = -1,
        display_cost: bool = True,
    ) -> None:
        """Init an OpenAI GPT/Codex engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            display_cost (bool, optional): Display cost of API call. Defaults to True.
        """
        assert (
            os.getenv("OPENAI_API_KEY", api_key) is not None
        ), "must pass on the api_key or set OPENAI_API_KEY in the environment"
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")

        # convert rate limit to minmum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)
        self.current_key_idx = 0
        self.display_cost = display_cost


class OpenAIChatEngine(OpenAIEngine):
    def __init__(
        self,
        model: str,
        api_key: Union[str, List[str], None] = None,
        stop: List[str] = ["\n\n"],
        rate_limit: int = -1,
        temperature: float = 0,
        display_cost: bool = True,
    ) -> None:
        """Init an OpenAI Chat engine

        Args:
            model (str): Model family.
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            temperature (int, optional): Defaults to 0.
            display_cost (bool, optional): Display cost of API call. Defaults to True.
        """
        self.stop = stop
        self.temperature = temperature

        self.model = model
        assert (
            model in supported_chat_models
        ), f"model must be one of {supported_chat_models}"

        self.cost_per_thousand_input_tokens = 0.0010  # $0.0010 per 1,000 input tokens
        self.cost_per_thousand_output_tokens = 0.0020  # $0.0020 per 1,000 output tokens

        super().__init__(api_key, rate_limit, display_cost)

    def calculate_cost(self, prompt_tokens, completion_tokens):

        # Calculate the costs
        input_cost = prompt_tokens * self.cost_per_thousand_input_tokens / 1000
        output_cost = completion_tokens * self.cost_per_thousand_output_tokens / 1000

        # Total cost
        total_cost = input_cost + output_cost
        return total_cost

    @backoff.on_exception(
        backoff.expo,
        (
            APIError,
            RateLimitError,
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            OpenAIError,
        ),
    )
    def generate(
        self,
        prompt: Union[str, list[dict]],
        max_new_tokens: int = 50,
        temperature: float = 0,
        **kwargs,
    ) -> List[str]:
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        start_time = time.time()
        if (
            self.request_interval > 0
            and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)
        client = OpenAI(api_key=self.api_keys[self.current_key_idx])
        if isinstance(prompt, str):
            messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            messages = prompt
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        if self.request_interval > 0:
            self.next_avil_time[self.current_key_idx] = (
                max(start_time, self.next_avil_time[self.current_key_idx])
                + self.request_interval
            )
        if self.display_cost:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost = self.calculate_cost(prompt_tokens, completion_tokens)

            logger.info(f"Prompt tokens: {prompt_tokens}")
            logger.info(f"Completion tokens: {completion_tokens}")
            logger.info(f"Total cost for this chat API call: ${cost:.4f}")
        return [choice.message.content for choice in response.choices]


class OpenAIEmbeddingEngine(OpenAIEngine):
    def __init__(
        self,
        api_key: Union[str, List[str], None] = None,
        rate_limit: int = -1,
        display_cost: bool = True,
    ) -> None:
        """Init an OpenAI Embedding engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            display_cost (bool, optional): Display cost of API call. Defaults to True.
        """
        self.model = "text-embedding-ada-002"
        self.cost_per_thousand_tokens = 0.0001

        super().__init__(api_key, rate_limit, display_cost)

    @backoff.on_exception(
        backoff.expo,
        (
            APIError,
            RateLimitError,
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            OpenAIError,
        ),
    )
    def embeddings(self, text: str) -> np.ndarray:
        client = OpenAI(api_key=self.api_keys[self.current_key_idx])
        response = client.embeddings.create(model=self.model, input=text)
        if self.display_cost:
            total_tokens = response.usage.total_tokens
            cost = self.cost_per_thousand_tokens * total_tokens / 1000
            logger.info(f"Total cost for this embedding API call: {cost}")
        return np.array([data.embedding for data in response.data])
