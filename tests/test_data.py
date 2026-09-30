"""Tests on fetched data. Set MUSELENS_DATA_DIR to a directory filled by ``muselens-data fetch``; skipped otherwise.

    MUSELENS_DATA_DIR=/path/to/data pytest tests/test_data.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from muselens_data import get_target, load_frozen_hst_mask, load_hst, prepare_line_window, target_ids, verify_target
from muselens_data.load import hst_variance

DATA = os.environ.get('MUSELENS_DATA_DIR')
pytestmark = pytest.mark.skipif(DATA is None, reason='MUSELENS_DATA_DIR not set')


@pytest.mark.parametrize('gid', target_ids())
def test_files_match_manifest(gid: str) -> None:
    assert verify_target(gid, Path(DATA)) == []


@pytest.mark.parametrize('gid', target_ids('phase1'))
def test_instrument_values_match_production(gid: str) -> None:
    """PSF, LSF, pixel scale and grid rotation rebuilt from the files equal the Phase 1 record."""
    p1 = get_target(gid)['phase1']
    lw = prepare_line_window(gid, Path(DATA))
    assert lw.psf.fwhm_arcsec == pytest.approx(p1['psf_fwhm_line'], abs=1e-6)
    assert lw.lsf_sigma_angstrom == pytest.approx(p1['lsf_sigma_angstrom'], abs=1e-6)
    assert lw.pixel_scale == pytest.approx(p1['muse_pixel_scale'], abs=1e-6)
    if p1['grid_rot_deg'] is not None:
        assert np.degrees(lw.grid_rotation_rad) == pytest.approx(p1['grid_rot_deg'], abs=1e-3)
    assert np.isfinite(lw.variance).any() and (lw.variance > 0).all()


@pytest.mark.parametrize('gid', target_ids('production')[:10])
def test_hst_variance(gid: str) -> None:
    hst = load_hst(gid, Path(DATA))
    assert hst.pixel_scale == pytest.approx(0.03, abs=1e-6)
    m = load_frozen_hst_mask(gid)
    var = hst_variance(hst, m)
    assert np.array_equal(np.isfinite(var), m.good_mask)
