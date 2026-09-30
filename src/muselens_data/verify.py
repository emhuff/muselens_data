"""Check downloaded files against the manifest of the files MUSElens used.

The manifest (data/manifest.json) was computed from the MUSElens working copies on 2026-09-30. For every FITS
file, every HDU's data array and header (less DATE/CHECKSUM/DATASUM) must match. Catalogues are compared by
whole-file hash, or, for VizieR TSV output whose header records the query date, by the hash of the data lines.
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from muselens_data.fetch import hst_paths, load_catalog_specs, muse_path
from muselens_data.fingerprint import data_lines_sha256, file_sha256, fits_fingerprint

__all__ = ['load_manifest', 'verify_target', 'verify_catalogs', 'require', 'VerificationError']


class VerificationError(RuntimeError):
    """Raised by ``require`` when a file is missing or differs from the manifest."""


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    with resources.files('muselens_data').joinpath('data/manifest.json').open() as f:
        return json.load(f)


def _compare(path: Path, expected: list[dict[str, Any]]) -> list[str]:
    if not path.exists():
        return [f'{path}: missing']
    got = fits_fingerprint(path)
    problems = []
    if [h['name'] for h in got] != [h['name'] for h in expected]:
        problems.append(f'{path.name}: HDUs {[h["name"] for h in got]} != {[h["name"] for h in expected]}')
    for g, e in zip(got, expected):
        for key in ('shape', 'dtype', 'data_sha256', 'header_sha256'):
            if g[key] != e[key]:
                problems.append(f'{path.name} [{e["name"]}] {key}: {g[key]} != expected {e[key]}')
    return problems


def verify_target(galaxy_id: str, data_dir: str | Path) -> list[str]:
    """List of discrepancies for one target (empty when HST science, weight and MUSE cube all match)."""
    gid = str(galaxy_id)
    m = load_manifest()['targets'].get(gid)
    if m is None:
        raise KeyError(f'{gid} not in the manifest')
    sci, wht = hst_paths(data_dir, gid)
    return (_compare(sci, m['hst_sci']['hdus']) + _compare(wht, m['hst_wht']['hdus'])
            + _compare(muse_path(data_dir, gid), m['muse']['hdus']))


def verify_catalogs(data_dir: str | Path) -> list[str]:
    problems = []
    for spec in load_catalog_specs():
        p = Path(data_dir) / 'catalogs' / spec['filename']
        if not p.exists():
            problems.append(f'{p}: missing')
            continue
        got = data_lines_sha256(p) if spec['hash'] == 'data_lines_sha256' else file_sha256(p)
        if got != spec['sha256']:
            problems.append(f'{p.name}: {spec["hash"]} {got} != expected {spec["sha256"]}')
    return problems


def require(problems: list[str]) -> None:
    """Raise VerificationError listing ``problems`` if there are any."""
    if problems:
        raise VerificationError('\n'.join(problems))
