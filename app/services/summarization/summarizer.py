"""LLM-based meeting summarization with decisions and action items."""

from pathlib import Path


class Summarizer:
    """Provider-agnostic LLM client producing structured Minutes of Meeting.

    TODO: implement in a later prompt.
    """

    async def summarize(self, audio_path: Path) -> object:
        """Return summary, decisions and action items."""
        raise NotImplementedError
