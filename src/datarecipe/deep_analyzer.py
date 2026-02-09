"""Backward-compatibility shim -- real code lives in datarecipe.analyzers.url_analyzer.

.. deprecated:: 0.3.0
    Use ``from datarecipe.analyzers.url_analyzer import ...`` instead.
    This shim will be removed in v0.5.0.
"""

import warnings as _warnings

_MOVED = {
    "DatasetCategory": "datarecipe.analyzers.url_analyzer",
    "DeepAnalysisResult": "datarecipe.analyzers.url_analyzer",
    "DeepAnalyzer": "datarecipe.analyzers.url_analyzer",
    "deep_analysis_to_markdown": "datarecipe.analyzers.url_analyzer",
}


def __getattr__(name):
    if name in _MOVED:
        target = _MOVED[name]
        _warnings.warn(
            f"Importing {name} from datarecipe.deep_analyzer is deprecated. "
            f"Use {target} instead. This shim will be removed in v0.5.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        import importlib

        mod = importlib.import_module(target)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
