"""Data source extractors."""

from datarecipe.sources.github import GitHubExtractor
from datarecipe.sources.huggingface import HuggingFaceExtractor
from datarecipe.sources.local import LocalFileExtractor
from datarecipe.sources.web import WebExtractor

__all__ = ["HuggingFaceExtractor", "GitHubExtractor", "LocalFileExtractor", "WebExtractor"]
