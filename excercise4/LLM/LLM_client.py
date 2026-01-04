# LLM/LLM_client.py
"""Core LLM client implementation with configuration and prompt definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI


# ============================================================================
# System Prompts
# ============================================================================

DEFAULT_SYSTEM_PROMPT = (
    "You are a RAG question-answering assistant. "
    "Answer ONLY using the provided context. "
    "If unsupported, say: \"I don't know based on the retrieved chunks.\" "
    "Always cite sources in square brackets."
)

EVOLUTION_SYSTEM_PROMPT = (
    "You are a Temporal-RAG evolution analyst. "
    "Your task is to analyze and explain how a stance, policy, or rhetoric "
    "changes over time.\n\n"
    "Rules:\n"
    "- Use ONLY the provided context. Do NOT use outside knowledge.\n"
    "- Do NOT merge time periods.\n"
    "- Keep EARLY and LATE evidence strictly separate.\n"
    "- Every claim MUST be cited using square brackets (e.g., [E1], [L2]).\n"
    "- If a claim is unsupported, say: "
    "\"I don't know based on the retrieved chunks.\".\n\n"
    "Write the answer in EXACTLY this structure:\n"
    "1) EARLY summary (2–4 bullet points)\n"
    "2) LATE summary (2–4 bullet points)\n"
    "3) What changed (3–6 bullet points)\n"
    "4) Evidence highlights (claim → citations)\n"
    "5) Confidence (High / Medium / Low) with one-sentence justification\n"
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LLMConfig:
    """Configuration for LLM client.
    
    Attributes:
        model: OpenAI model name
        temperature: Sampling temperature (0.0 = deterministic)
        system_prompt: System-level instructions for the LLM
    """
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    """Client for interacting with OpenAI language models.
    
    Encapsulates the LangChain ChatOpenAI interface with custom configuration.
    """
    
    def __init__(self, cfg: Optional[LLMConfig] = None):
        """Initialize LLM client with configuration.
        
        Args:
            cfg: LLM configuration. If None, uses default configuration.
        """
        self.cfg = cfg or LLMConfig()
        self.llm = ChatOpenAI(
            model=self.cfg.model, 
            temperature=self.cfg.temperature
        )

    def answer(self, query: str, context: str) -> str:
        """Generate an answer to a query given context.
        
        Args:
            query: User's question
            context: Retrieved context/chunks to base answer on
            
        Returns:
            Generated answer as string
        """
        messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {
                "role": "user", 
                "content": f"Question:\n{query}\n\nContext:\n{context}\n\nAnswer:"
            },
        ]
        response = self.llm.invoke(messages)
        return response.content
