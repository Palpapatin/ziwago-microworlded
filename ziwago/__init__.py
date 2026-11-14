"""NEUROWEAVE v10.0 «Genesis» — виртуальная биология по GENESIS_MANIFESTO."""
from __future__ import annotations

__version__ = "10.0-genesis"

# GenesisOrganism/GenesisSystem (мёртвый unified-стек) архивированы в _archive/
# (класс B, §6/§8.7); живой организм — organism.integrated.IntegratedOrganism.


def GenesisConfig(*args, **kwargs):  # noqa: N802
    from .config import GenesisConfig as _GC
    return _GC(*args, **kwargs)


__all__ = ["GenesisConfig", "__version__"]
