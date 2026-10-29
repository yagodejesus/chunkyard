"""chunkyard — lint your RAG corpus: duplicates, near-duplicates, boilerplate, broken encoding and badly sized chunks."""

__version__ = "0.1.0"

from .lint import CODES, Chunk, Finding, Report, approx_tokens, fix, from_records, lint, load_chunks

__all__ = ["__version__", "CODES", "Chunk", "Finding", "Report", "approx_tokens", "fix", "from_records", "lint", "load_chunks"]
