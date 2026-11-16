from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.organism_core import ZiwagoOrganism


class _Dreamer:
    def __init__(self):
        self.graphs = {}

    def register_graph(self, name, edges, node_types):
        self.graphs[name] = {"edges": edges, "node_types": node_types}

    def match_graphs(self, source, target):
        return None


def test_rem_dream_applies_node_cap_to_nodes_not_edge_count():
    organism = ZiwagoOrganism.__new__(ZiwagoOrganism)
    organism.world_graph = SimpleNamespace(edges=[
        SimpleNamespace(
            subject=f"subject-{index}",
            target=f"target-{index}",
            relation="related",
            source="family-a",
            status="asserted",
        )
        for index in range(10)
    ])
    organism.isomorphism_dreamer = _Dreamer()

    report = organism._rem_dream()

    graph = organism.isomorphism_dreamer.graphs["family-a"]
    assert len(graph["node_types"]) <= report["node_cap"] == 6
    assert len(graph["edges"]) == 3
    assert report["edges_skipped_by_node_cap"] == 7
    assert report["graphs_registered"] == 1
