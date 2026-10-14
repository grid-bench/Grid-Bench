"""Manuscript display identities, separate from the frozen analysis registry."""

NAMES = {
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "claude-sonnet": "Claude 5 Sonnet",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "zai.glm-5": "GLM-5",
    "moonshotai.kimi-k2.5": "Kimi K2.5",
    "minimax.minimax-m2.5": "MiniMax M2.5",
}

WRAPPED_NAMES = {
    "gpt-5.6-sol": "GPT-5.6\nSol",
    "claude-sonnet": "Claude 5\nSonnet",
    "gpt-5.4-mini": "GPT-5.4\nMini",
    "zai.glm-5": "GLM-5",
    "moonshotai.kimi-k2.5": "Kimi\nK2.5",
    "minimax.minimax-m2.5": "MiniMax\nM2.5",
}


def tex_header(model: str) -> str:
    """Wrap full identities at natural word boundaries in narrow table headers."""
    return r"\shortstack{" + WRAPPED_NAMES[model].replace("\n", r"\\") + "}"
