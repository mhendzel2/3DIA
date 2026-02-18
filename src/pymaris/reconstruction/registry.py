"""Registry for reconstruction plugins."""

from __future__ import annotations

from dataclasses import dataclass, field

from pymaris.reconstruction.plugin import ReconstructionPlugin
from pymaris.reconstruction.types import ReconstructionPluginInfo


@dataclass
class ReconstructionRegistry:
    """In-memory plugin registry with capability filtering."""

    plugins: dict[str, ReconstructionPlugin] = field(default_factory=dict)

    def register(self, plugin: ReconstructionPlugin) -> None:
        name = plugin.info.name
        if not name:
            raise ValueError("reconstruction plugin must define a non-empty info.name")
        self.plugins[name] = plugin

    def get(self, name: str) -> ReconstructionPlugin:
        key = str(name)
        if key not in self.plugins:
            available = ", ".join(sorted(self.plugins.keys()))
            raise KeyError(f"reconstruction plugin not found: {key!r} (available: {available})")
        return self.plugins[key]

    def list_info(
        self,
        *,
        modality: str | None = None,
        ndim: int | None = None,
    ) -> list[ReconstructionPluginInfo]:
        normalized = str(modality).strip().lower() if modality is not None else None
        infos: list[ReconstructionPluginInfo] = []
        for plugin in self.plugins.values():
            info = plugin.info
            if normalized:
                tags = {value.lower() for value in info.modality_tags}
                if normalized not in tags:
                    continue
            if ndim is not None and info.supported_dims and ndim not in info.supported_dims:
                continue
            infos.append(info)
        return sorted(infos, key=lambda item: item.name)


DEFAULT_RECONSTRUCTION_REGISTRY = ReconstructionRegistry()
