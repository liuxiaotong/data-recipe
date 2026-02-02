"""Provider 插件系统

通过 entry_points 机制发现和加载已安装的 providers。
"""

import importlib.metadata
from typing import Optional

from datarecipe.schema import DeploymentProvider


class ProviderNotFoundError(Exception):
    """Provider 未找到"""
    pass


def discover_providers() -> dict[str, type]:
    """发现所有已安装的 providers

    Returns:
        dict: provider 名称 -> provider 类的映射
    """
    providers = {}

    try:
        eps = importlib.metadata.entry_points(group="datarecipe.providers")
        for ep in eps:
            try:
                provider_class = ep.load()
                providers[ep.name] = provider_class
            except Exception as e:
                print(f"Warning: Failed to load provider {ep.name}: {e}")
    except Exception:
        pass

    # 确保 local provider 始终可用
    if "local" not in providers:
        from datarecipe.providers.local import LocalFilesProvider
        providers["local"] = LocalFilesProvider

    return providers


def get_provider(name: str) -> DeploymentProvider:
    """获取指定名称的 provider 实例

    Args:
        name: provider 名称

    Returns:
        DeploymentProvider 实例

    Raises:
        ProviderNotFoundError: 如果 provider 未找到
    """
    providers = discover_providers()

    if name not in providers:
        available = list(providers.keys())
        raise ProviderNotFoundError(
            f"Provider '{name}' not found. "
            f"Available providers: {available}. "
            f"Install with: pip install datarecipe-{name}"
        )

    return providers[name]()


def list_providers() -> list[dict]:
    """列出所有可用的 provider 信息

    Returns:
        list: provider 信息列表
    """
    providers = discover_providers()
    result = []

    for name, provider_class in providers.items():
        try:
            instance = provider_class()
            result.append({
                "name": name,
                "description": instance.description,
            })
        except Exception as e:
            result.append({
                "name": name,
                "description": f"(Error loading: {e})",
            })

    return result


__all__ = [
    "ProviderNotFoundError",
    "discover_providers",
    "get_provider",
    "list_providers",
]
