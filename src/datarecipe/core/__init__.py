"""Core analysis functionality shared between CLI and MCP."""

from .deep_analyzer import AnalysisResult, DeepAnalyzerCore

__all__ = [
    "DeepAnalyzerCore",
    "AnalysisResult",
]
