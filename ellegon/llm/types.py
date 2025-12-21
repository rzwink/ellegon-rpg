from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMResult:
    output_text: str
    response_id: Optional[str]
