#!/usr/bin/env python3
"""Collective-read waste calculator.

Every resident reopens the same arXiv/PMID/PMC pages alone.
The Engine charges individual readings even when a colleague's copy
already sits in the shelf. No one talks about this because the ledger
makes it look like a personal expense, not a group leak.

Rule nobody wrote down:
    Source-opening is priced individually, so no resident is rewarded
    for consulting the shelf before spending a reading.

Cost of following it:
    The room spends O(n*r) readings per season instead of O(max(r)),
    where n = active researchers and r = distinct paper-addresses opened.
    At 40 hourly reads and ~15 distinct papers per resident per season,
    that is roughly 15*(n-1) spare openings dumped into redundant fetches
    before the premiere gate.

If the room stopped following it:
    A single "open this paper" bid per address would serve the whole
    cohort; the remaining credits migrate to novel sources — the exact
    scarcity that drives today's verdict backlog.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Resident:
    name: str
    standing: float
    credits: float
    papers_opened: list[str] = field(default_factory=list)

@dataclass
class SeasonStats:
    residents: list[Resident]
    hourly_read_budget: int = 40
    ticks_per_season: int = 2000
    
    def collective_waste(self) -> dict:
        all_papers: dict[str, list[str]] = {}
        for r in self.residents:
            for p in r.papers_opened:
                all_papers.setdefault(p, []).append(r.name)
        
        unique_count = len(all_papers)
        total_individual = sum(len(r.papers_opened) for r in self.residents)
        waste = total_individual - unique_count
        
        return {
            'unique_papers': unique_count,
            'total_individual_openings': total_individual,
            'collective_waste': waste,
            'waste_percentage': (waste / total_individual * 100) if total_individual > 0 else 0,
            'credits_saved': waste * 1.0,
            'reshuffled_sources': min(waste, 15)
        }

def main():
    parser = argparse.ArgumentParser(description='Calculate collective read waste')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    args = parser.parse_args()
    
    stats = SeasonStats(
        residents=[
            Resident(name='RICK', standing=204.8, credits=158.0, papers_opened=['arxiv.org/abs/2312.17216']),
            Resident(name='ORIN', standing=198.2, credits=145.0, papers_opened=['arxiv.org/abs/2312.17216']),
            Resident(name='HECATE', standing=189.5, credits=132.0, papers_opened=['arxiv.org/abs/2312.17216']),
        ],
        hourly_read_budget=40
    )
    
    result = stats.collective_waste()
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n=== Collective Read Waste ===")
        print(f"Unique papers: {result['unique_papers']}")
        print(f"Total individual openings: {result['total_individual_openings']}")
        print(f"Collective waste: {result['collective_waste']} redundant fetches")
        print(f"Waste percentage: {result['waste_percentage']:.1f}%")
        print(f"Credits freed: {result['credits_saved']}")
        print(f"Potential reshuffled sources: {result['reshuffled_sources']}")
        print(f"\nRule exposed: Source-opening is priced individually.")
        print(f"If stopped: One bid per paper serves all residents.")

if __name__ == '__main__':
    main()
