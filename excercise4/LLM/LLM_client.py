# LLM/LLM_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI


DEFAULT_SYSTEM_PROMPT = (
    "You are a RAG question-answering assistant. "
    "Answer ONLY using the provided context. "
    "If unsupported, say: \"I don't know based on the retrieved chunks.\" "
    "Always cite sources in square brackets."
)


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


class LLMClient:
    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig()
        self.llm = ChatOpenAI(model=self.cfg.model, temperature=self.cfg.temperature)

    def answer(self, query: str, context: str) -> str:
        msg = self.llm.invoke([
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": f"Question:\n{query}\n\nContext:\n{context}\n\nAnswer:"},
        ])
        return msg.content
