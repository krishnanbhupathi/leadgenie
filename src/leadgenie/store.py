"""Idempotency store — re-runs never reprocess (or re-pay for) a lead already handled.

A lead ID is marked done only after its result is written, and the store is flushed to
disk immediately, so a crash mid-run resumes exactly where it left off.
"""
import json
import os
from typing import Set


class ProcessedStore:
    def __init__(self, path: str = ".processed.json"):
        self.path = path
        self._ids: Set[str] = set()
        if os.path.exists(path):
            with open(path) as f:
                self._ids = set(json.load(f))

    def __contains__(self, lead_id: str) -> bool:
        return lead_id in self._ids

    def add(self, lead_id: str) -> None:
        self._ids.add(lead_id)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sorted(self._ids), f)
        os.replace(tmp, self.path)
