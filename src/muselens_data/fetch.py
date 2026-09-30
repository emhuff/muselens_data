"""Download the MUSElens data from the public archives, with the exact requests used by MUSElens.

Layout under ``data_dir``::

    hst/{gid}_f814w_sci.fits, hst/{gid}_f814w_wht.fits   HLA fitscut, CANDELS GOODS-S ACS F814W v1.0
    muse/muse_cutout_{gid}.fits                           MUSE-Wide cutout API, or converted HUDF bundle
    hudf_sources/DR2_{DATASET}_{ID:06d}.fits              AMUSED per-source MPDAF bundles (HUDF only)
    catalogs/...                                          source catalogues (see data/catalogs.json)

Existing files are kept (use ``verify`` to check them). A download is written to a temporary name and renamed
only after it parses as FITS, so an interrupted run never leaves a truncated file under the final name.
"""
from __future__ import annotations

import json
import logging
import time
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import requests
from astropy.io import fits

from muselens_data.constants import (AMUSED_SOURCE_URL, CANDELS_F814W_SCI, CANDELS_F814W_WHT, HLA_FITSCUT_URL,
                                     HST_PIXEL_SCALE, MUSEWIDE_CUTOUT_URL)
from muselens_data.targets import get_target

__all__ = ['hst_paths', 'muse_path', 'hudf_source_path', 'fetch_hst', 'fetch_muse', 'fetch_target',
           'fetch_catalogs', 'convert_hudf_source', 'load_catalog_specs']

logger = logging.getLogger(__name__)


def hst_paths(data_dir: str | Path, gid: str) -> tuple[Path, Path]:
    d = Path(data_dir) / 'hst'
    return d / f'{gid}_f814w_sci.fits', d / f'{gid}_f814w_wht.fits'


def muse_path(data_dir: str | Path, gid: str) -> Path:
    return Path(data_dir) / 'muse' / f'muse_cutout_{gid}.fits'


def hudf_source_path(data_dir: str | Path, target: dict[str, Any]) -> Path:
    return Path(data_dir) / 'hudf_sources' / target['request']['muse']['source_file']


def _download(url: str, dest: Path, timeout: float = 300.0, retries: int = 3, params: dict | None = None) -> None:
    """Stream ``url`` to ``dest`` via a temporary file. Raises after ``retries`` failed attempts."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.part')
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, params=params, stream=True, timeout=timeout) as r:
                if r.status_code != 200:
                    raise RuntimeError(f'HTTP {r.status_code} for {r.url}: {r.text[:200]!r}')
                with open(tmp, 'wb') as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return
        except (requests.RequestException, RuntimeError, OSError) as e:
            last = e
            if tmp.exists():
                tmp.unlink()
            logger.warning('download attempt %d/%d failed for %s: %s', attempt + 1, retries, url, e)
            time.sleep(5.0 * (attempt + 1))
    raise RuntimeError(f'download failed after {retries} attempts: {url}: {last}') from last


def _check_fits(path: Path, min_hdus: int, label: str) -> None:
    try:
        with fits.open(path) as h:
            if len(h) < min_hdus:
                raise RuntimeError(f'{label}: {len(h)} HDUs, expected >= {min_hdus}')
    except OSError as e:
        bad = path.with_name(path.name + '.bad')
        path.replace(bad)
        raise RuntimeError(f'{label}: response is not FITS (kept as {bad}): {e}') from e


def fetch_hst(target: dict[str, Any], data_dir: str | Path, timeout: float = 120.0) -> tuple[Path, Path]:
    """HLA fitscut science (drz) and weight (wht) cutouts, square, ``request['hst']['size']`` pixels at 0.03"."""
    gid = target['galaxy_id']
    req = target['request']['hst']
    sci, wht = hst_paths(data_dir, gid)
    for dataset, dest in ((CANDELS_F814W_SCI, sci), (CANDELS_F814W_WHT, wht)):
        if dest.exists():
            continue
        # RA/Dec formatted with str(), exactly as MUSElens built the URL
        url = HLA_FITSCUT_URL.format(dataset=dataset, ra=req['ra'], dec=req['dec'], size=req['size'],
                                     pixel_scale=HST_PIXEL_SCALE)
        _download(url, dest, timeout=timeout)
        _check_fits(dest, 1, f'{gid} {dataset}')
        with fits.open(dest) as h:
            d = h[0].data
            if d is None or not np.isfinite(d).any() or np.all(d == 0):
                raise RuntimeError(f'{gid} {dataset}: empty cutout (outside the mosaic footprint?)')
    return sci, wht


def convert_hudf_source(source_path: Path, out_path: Path) -> None:
    """Rewrite an AMUSED MPDAF bundle in the MUSE-Wide cutout layout (verbatim port of MUSE-Wide
    ``src/hudf_data.convert_hudf_source``): PRIMARY with ORIGFILE and the FSF* cards, then DATA_DCBGC and
    EFF_STAT as float32 with the CUB_MUSE_CUBE_{DATA,STAT} headers."""
    with fits.open(source_path) as h:
        data_hdu = h['CUB_MUSE_CUBE_DATA']
        stat_hdu = h['CUB_MUSE_CUBE_STAT']
        for name, hdu in (('DATA', data_hdu), ('STAT', stat_hdu)):
            if hdu.data.ndim != 3:
                raise ValueError(f'{source_path.name} CUB_MUSE_CUBE_{name}: ndim {hdu.data.ndim} != 3')
        for key in ('CRVAL3', 'CD3_3', 'CTYPE3'):
            if key not in data_hdu.header:
                raise ValueError(f'{source_path.name}: missing {key}')
        primary = fits.PrimaryHDU()
        primary.header['ORIGFILE'] = (source_path.name, 'AMUSED source file')
        for card in h[0].header.cards:
            if card.keyword.startswith('FSF'):
                primary.header[card.keyword] = (card.value, card.comment)
        data_out = fits.ImageHDU(data=data_hdu.data.astype(np.float32), header=data_hdu.header)
        data_out.name = 'DATA_DCBGC'
        stat_out = fits.ImageHDU(data=stat_hdu.data.astype(np.float32), header=stat_hdu.header)
        stat_out.name = 'EFF_STAT'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([primary, data_out, stat_out]).writeto(out_path, overwrite=True)


def fetch_muse(target: dict[str, Any], data_dir: str | Path, timeout: float = 600.0) -> Path:
    """MUSE cube for one target: a MUSE-Wide API cutout, or a HUDF AMUSED bundle converted to that layout."""
    gid = target['galaxy_id']
    req = target['request']['muse']
    dest = muse_path(data_dir, gid)
    if dest.exists():
        return dest
    if req['service'] == 'musewide_api':
        url = MUSEWIDE_CUTOUT_URL.format(ra=req['ra'], dec=req['dec'], box=req['box_arcsec'], l0=req['l0'],
                                         l1=req['l1'])
        _download(url, dest, timeout=timeout)
        _check_fits(dest, 3, f'{gid} MUSE-Wide cutout')
    elif req['service'] == 'amused':
        src = hudf_source_path(data_dir, target)
        if not src.exists():
            _download(AMUSED_SOURCE_URL.format(name=req['source_file']), src, timeout=timeout)
            _check_fits(src, 3, f'{gid} AMUSED bundle')
        convert_hudf_source(src, dest)
    else:
        raise ValueError(f'{gid}: unknown MUSE service {req["service"]!r}')
    return dest


def fetch_target(galaxy_id: str, data_dir: str | Path) -> dict[str, Path]:
    t = get_target(galaxy_id)
    sci, wht = fetch_hst(t, data_dir)
    return {'hst_sci': sci, 'hst_wht': wht, 'muse': fetch_muse(t, data_dir)}


def load_catalog_specs() -> list[dict[str, Any]]:
    with resources.files('muselens_data').joinpath('data/catalogs.json').open() as f:
        return json.load(f)


def fetch_catalogs(data_dir: str | Path) -> dict[str, Path]:
    """Download every catalogue listed in data/catalogs.json into ``data_dir/catalogs``."""
    out = {}
    for spec in load_catalog_specs():
        dest = Path(data_dir) / 'catalogs' / spec['filename']
        if not dest.exists():
            _download(spec['url'], dest, params=spec.get('params'))
        out[spec['name']] = dest
    return out
