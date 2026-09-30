"""Command line: fetch, verify, and check regenerated masks.

    muselens-data fetch   DATA_DIR [--subset all|phase1|production] [--ids ID ...] [--no-catalogs] [--no-verify]
    muselens-data verify  DATA_DIR [--subset ...] [--ids ...] [--no-catalogs]
    muselens-data masks   DATA_DIR [--subset phase1|production] [--ids ...]

Exit status is non-zero if any file fails to download, differs from the manifest, or any regenerated mask
differs from the frozen production mask. ``fetch --no-verify`` skips the manifest comparison: use it when the
archives have legitimately updated a product and the newer data are wanted. Data fetched that way are not the
data MUSElens used, and the frozen HST masks may no longer apply to them (regenerate them with ``masks``).
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from muselens_data.fetch import fetch_catalogs, fetch_target
from muselens_data.load import regenerate_hst_mask
from muselens_data.masks import load_frozen_hst_mask
from muselens_data.targets import target_ids
from muselens_data.verify import verify_catalogs, verify_target


def _ids(a: argparse.Namespace) -> list[str]:
    return list(a.ids) if a.ids else target_ids(a.subset)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='muselens-data')
    ap.add_argument('command', choices=['fetch', 'verify', 'masks'])
    ap.add_argument('data_dir')
    ap.add_argument('--subset', default='all', choices=['all', 'phase1', 'production'])
    ap.add_argument('--ids', nargs='*')
    ap.add_argument('--no-catalogs', action='store_true')
    ap.add_argument('--no-verify', action='store_true',
                    help='fetch only: skip the checksum comparison against the MUSElens manifest')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    failures: dict[str, str] = {}
    ids = _ids(a)
    if a.no_verify and a.command != 'fetch':
        ap.error('--no-verify applies to fetch only')
    check = not a.no_verify
    if not check:
        logging.warning('checksum verification OFF: fetched files are not compared with the MUSElens manifest')
    if a.command in ('fetch', 'verify') and not a.no_catalogs:
        try:
            if a.command == 'fetch':
                fetch_catalogs(a.data_dir)
            probs = verify_catalogs(a.data_dir) if check else []
            if probs:
                failures['catalogs'] = '; '.join(probs)
        except Exception as e:
            failures['catalogs'] = f'{type(e).__name__}: {e}'
    for i, gid in enumerate(ids):
        try:
            if a.command == 'fetch':
                fetch_target(gid, a.data_dir)
            if a.command == 'verify' or (a.command == 'fetch' and check):
                probs = verify_target(gid, a.data_dir)
                if probs:
                    failures[gid] = '; '.join(probs)
            else:
                frozen = load_frozen_hst_mask(gid)
                new = regenerate_hst_mask(gid, a.data_dir)
                diff = []
                if not np.array_equal(frozen.segmap, new.segmap):
                    diff.append(f'segmap differs in {int((frozen.segmap != new.segmap).sum())} px')
                if not np.array_equal(frozen.good_mask, new.good_mask):
                    diff.append(f'good_mask differs in {int((frozen.good_mask != new.good_mask).sum())} px')
                if frozen.target_label != new.target_label:
                    diff.append(f'target_label {new.target_label} != {frozen.target_label}')
                if diff:
                    failures[gid] = '; '.join(diff)
        except Exception as e:
            failures[gid] = f'{type(e).__name__}: {e}'
        status = 'FAIL ' + failures[gid] if gid in failures else 'ok'
        print(f'[{i + 1}/{len(ids)}] {gid} {status}', flush=True)
    print(('' if check else '[unverified] ') + f'{a.command}: {len(ids) - sum(g != "catalogs" for g in failures)}/{len(ids)} targets ok'
          + ('' if 'catalogs' not in failures else f'; catalogs FAILED: {failures["catalogs"]}'))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
