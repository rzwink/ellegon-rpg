from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from ellegon.llm.types import LLMResult


class LLMGateway(Protocol):
    def generate(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        previous_response_id: Optional[str] = None,
    ) -> LLMResult:
        ...


@dataclass
class OpenAIGateway:
    client: Optional[object] = None

    def _client(self) -> object:
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI()
        return self.client

    def generate(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        previous_response_id: Optional[str] = None,
    ) -> LLMResult:
        kwargs = {}
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        response = self._client().responses.create(
            model=model,
            instructions=instructions,
            input=input_text,
            **kwargs,
        )
        output_text = response.output_text.strip()
        response_id = getattr(response, "id", None)
        return LLMResult(output_text=output_text, response_id=response_id)


@dataclass
class FakeGateway:
    output_text: str = "FAKE_DM_RESPONSE"
    response_id: str = "fake-response-id"

    def generate(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        previous_response_id: Optional[str] = None,
    ) -> LLMResult:
        return LLMResult(output_text=self.output_text, response_id=self.response_id)
