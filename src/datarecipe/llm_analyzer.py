"""Backward-compatibility shim -- real code lives in datarecipe.analyzers.llm_url_analyzer.

.. deprecated:: 0.3.0
    Use ``from datarecipe.analyzers.llm_url_analyzer import ...`` instead.
    This shim will be removed in v0.5.0.
"""

import warnings as _warnings

_MOVED = {
    "LLMAnalyzer": "datarecipe.analyzers.llm_url_analyzer",
    "MultiSourceContent": "datarecipe.analyzers.llm_url_analyzer",
    "ANALYSIS_PROMPT": "datarecipe.analyzers.llm_url_analyzer",
    # Re-exported from deep_analyzer in the original module
    "DatasetCategory": "datarecipe.analyzers.url_analyzer",
    "DeepAnalysisResult": "datarecipe.analyzers.url_analyzer",
    "DeepAnalyzer": "datarecipe.analyzers.url_analyzer",
}


def __getattr__(name):
    if name in _MOVED:
        target = _MOVED[name]
        _warnings.warn(
            f"Importing {name} from datarecipe.llm_analyzer is deprecated. "
            f"Use {target} instead. This shim will be removed in v0.5.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        import importlib

        mod = importlib.import_module(target)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
