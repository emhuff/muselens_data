"""The frozen MUSElens target table (366 targets: 290 MUSE-Wide DR1, 76 MUSE HUDF DR2).

Each record is the MUSElens pool record (``experiments/pool/pool.json``) plus:

    request      the exact archive requests that reproduce the cutouts
    phase1       production Phase 1 (tag poolpsf) instrument values, or None where no fit exists (8 targets)
    in_phase1    True for the 358 targets with a production Phase 1 fit (and a frozen HST mask)
    production   True for the 128 rotators of the production population run
                 (experiments/population/runs/shear_hann_poolpsf_th8)

Selection flags are informational; nothing in this package applies them.
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

__all__ = ['load_targets', 'get_target', 'target_ids']


@lru_cache(maxsize=1)
def _table() -> dict[str, dict[str, Any]]:
    with resources.files('muselens_data').joinpath('data/targets.json').open() as f:
        rows = json.load(f)
    table = {r['galaxy_id']: r for r in rows}
    if len(table) != len(rows):
        raise ValueError('duplicate galaxy_id in targets.json')
    return table


def load_targets() -> list[dict[str, Any]]:
    """All target records (copies)."""
    return [json.loads(json.dumps(r)) for r in _table().values()]


def get_target(galaxy_id: str | int) -> dict[str, Any]:
    """One target record (a copy). Raises KeyError for an unknown id."""
    gid = str(galaxy_id)
    if gid not in _table():
        raise KeyError(f'unknown galaxy_id {gid}')
    return json.loads(json.dumps(_table()[gid]))


def target_ids(subset: str = 'all') -> list[str]:
    """Galaxy ids: 'all' (366), 'phase1' (358) or 'production' (128)."""
    if subset == 'all':
        return list(_table())
    if subset == 'phase1':
        return [g for g, r in _table().items() if r['in_phase1']]
    if subset == 'production':
        return [g for g, r in _table().items() if r['production']]
    raise ValueError(f"unknown subset {subset!r}; use 'all', 'phase1' or 'production'")
