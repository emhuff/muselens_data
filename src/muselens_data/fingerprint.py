"""Content fingerprints of FITS files and catalogue files.

A FITS fingerprint is, per HDU, the SHA-256 of the data array (dtype string, shape and raw bytes as stored) and
of the header cards (keyword and value) excluding the cards the archive services rewrite on every request
(DATE, CHECKSUM, DATASUM). Two files with equal fingerprints carry identical pixels and identical metadata;
the whole-file hash is recorded as well but is not required to match (the HLA cutout service stamps DATE).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

__all__ = ['VOLATILE_CARDS', 'array_sha256', 'header_sha256', 'fits_fingerprint', 'file_sha256',
           'data_lines_sha256']

VOLATILE_CARDS: frozenset[str] = frozenset({'DATE', 'CHECKSUM', 'DATASUM'})


def array_sha256(a: np.ndarray | None) -> str | None:
    if a is None:
        return None
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(repr(tuple(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def header_sha256(header: fits.Header) -> str:
    h = hashlib.sha256()
    for card in header.cards:
        if card.keyword in VOLATILE_CARDS:
            continue
        h.update(f'{card.keyword}={card.value!r}\n'.encode())
    return h.hexdigest()


def fits_fingerprint(path: str | Path) -> list[dict[str, Any]]:
    """Per-HDU fingerprint: name, shape, dtype, data and header hashes."""
    out = []
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            d = hdu.data
            out.append({'name': hdu.name,
                        'shape': None if d is None else list(d.shape),
                        'dtype': None if d is None else d.dtype.str,
                        'data_sha256': array_sha256(d),
                        'header_sha256': header_sha256(hdu.header)})
    return out


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def data_lines_sha256(path: str | Path, comment: str = '#') -> str:
    """Hash of the non-comment, non-blank lines of a text table (VizieR TSV headers carry a request date)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for line in f:
            s = line.rstrip(b'\r\n')
            if not s.strip() or s.startswith(comment.encode()):
                continue
            h.update(s + b'\n')
    return h.hexdigest()
