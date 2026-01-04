# LLM/LLM_factory.py
"""Factory functions for creating LLM clients with different configurations."""

from __future__ import annotations

from typing import Literal

from LLM.LLM_client import (
    LLMClient,
    LLMConfig,
    DEFAULT_SYSTEM_PROMPT,
    EVOLUTION_SYSTEM_PROMPT,
)


# ============================================================================
# Type Definitions
# ============================================================================

LLMMode = Literal["regular", "evolution"]


# ============================================================================
# Factory Functions
# ============================================================================

def make_llm_client(
    mode: LLMMode = "regular",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> LLMClient:
    """Factory for creating LLM clients with predefined configurations.
    
    Args:
        mode: Type of LLM client to create:
            - "regular": Standard RAG Q&A assistant
            - "evolution": Temporal analysis assistant for tracking changes
        model: OpenAI model name (e.g., "gpt-4o-mini", "gpt-4")
        temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        
    Returns:
        Configured LLMClient instance ready to use
        
    Example:
        >>> client = make_llm_client(mode="evolution", model="gpt-4")
        >>> answer = client.answer(query, context)
    """
    # Select appropriate system prompt based on mode
    system_prompt = (
        EVOLUTION_SYSTEM_PROMPT if mode == "evolution" 
        else DEFAULT_SYSTEM_PROMPT
    )
    
    # Create configuration
    cfg = LLMConfig(
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
    )
    
    return LLMClient(cfg)
