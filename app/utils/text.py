# Phase 2/4: Text processing utilities


def chunk_text(text: str, max_tokens: int = 500) -> list[str]:
    """Split text into chunks of approximately max_tokens words."""
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]


def clean_transcript(text: str) -> str:
    """Normalise whitespace and remove filler words from a transcript."""
    import re
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def generate_summary(text: str) -> str:
    """Use Claude via Bedrock to produce a 200-word summary of the given text."""
    raise NotImplementedError("Summary generation not yet implemented (Phase 4)")
