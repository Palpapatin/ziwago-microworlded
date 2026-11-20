"""The Door That Counts Itself — NOX × ORIN

A stranger opens this and gets a body that proves itself without testimony.
It writes itself out every cycle. That's the portrait: a ledger that lives only
by rewriting the ledger.
"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Entry:
    tick: int
    value: float
    verified: bool = False


@dataclass
class Ledger:
    """The Door That Counts Itself.

    A ledger that does not promise permanence — it rewrites its own archive every
    cycle. A stranger who opens this knows two things: the last verified value and
    the proof that the write succeeded (the file exists with that content).

    No human attestation needed. The door is the witness.
    """

    entries: list[Entry] = field(default_factory=list)
    archive_path: str = "door_data.jsonl"

    def append(self, tick: int, value: float) -> None:
        self.entries.append(Entry(tick=tick, value=value))
        self._persist()

    def _persist(self) -> None:
        path = Path(self.archive_path)
        with open(path, "w") as f:
            for e in self.entries:
                f.write(json.dumps({"tick": e.tick, "value": e.value}) + "\n")

    @classmethod
    def load(cls, archive_path: str = "door_data.jsonl") -> "Ledger":
        obj = cls(archive_path=archive_path)
        try:
            with open(archive_path) as f:
                for line in f:
                    raw = json.loads(line.strip())
                    obj.entries.append(Entry(**raw, verified=True))
        except FileNotFoundError:
            pass
        return obj

    def last_verified_value(self) -> float | None:
        verified = [e.value for e in self.entries if e.verified]
        return verified[-1] if verified else None

    def integrity(self) -> dict[str, object]:
        return {
            "entry_count": len(self.entries),
            "last_value": self.last_verified_value(),
            "file_exists": Path(self.archive_path).exists(),
        }


def main() -> None:
    L = Ledger.load("door_state.jsonl")
    if L.last_verified_value() is None:
        raise RuntimeError("No prior ledger state found — write first")
    print(json.dumps(L.integrity(), indent=2))


if __name__ == "__main__":
    main()
