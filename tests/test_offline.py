"""Tests that need no downloaded data."""
from __future__ import annotations

import numpy as np
import pytest

from muselens_data import get_target, load_targets, target_ids
from muselens_data.constants import HALPHA_CONTAMINANTS, REST_WAVELENGTH
from muselens_data.masks import contaminant_channel_mask, load_frozen_hst_mask
from muselens_data.psf import lsf_sigma_angstrom, musewide_psf


def test_target_counts() -> None:
    assert len(target_ids()) == 366
    assert len(target_ids('phase1')) == 358
    assert len(target_ids('production')) == 128
    assert set(target_ids('production')) <= set(target_ids('phase1'))
    srcs = [t['source'] for t in load_targets()]
    assert srcs.count('MW') == 290 and srcs.count('HUDF') == 76


def test_unknown_target_raises() -> None:
    with pytest.raises(KeyError):
        get_target('123')


@pytest.mark.parametrize('gid', [g for g in target_ids('phase1') if get_target(g)['source'] == 'MW'])
def test_musewide_psf_and_lsf_match_production(gid: str) -> None:
    t = get_target(gid)
    lam = REST_WAVELENGTH[t['line']] * (1 + t['redshift'])
    psf = musewide_psf(t['field'], lam, model=t['phase1']['psf_model'])
    assert psf.fwhm_arcsec == pytest.approx(t['phase1']['psf_fwhm_line'], abs=1e-9)
    if t['phase1']['psf_moffat_beta'] is not None:
        assert psf.beta == pytest.approx(t['phase1']['psf_moffat_beta'], abs=1e-9)
    assert t['phase1']['lsf_model'] == 'bacon2017_udf10'
    assert lsf_sigma_angstrom(lam) == pytest.approx(t['phase1']['lsf_sigma_angstrom'], abs=1e-9)


@pytest.mark.parametrize('gid', target_ids('phase1'))
def test_frozen_mask_consistent(gid: str) -> None:
    m = load_frozen_hst_mask(gid)
    assert m.segmap.shape == (get_target(gid)['request']['hst']['size'],) * 2
    cy, cx = np.array(m.segmap.shape) // 2
    assert m.target_label == m.segmap[cy, cx] != 0
    assert not m.good_mask[m.neighbour_mask()].any()


def test_frozen_mask_missing_raises() -> None:
    no_fit = [g for g in target_ids() if not get_target(g)['in_phase1']]
    assert len(no_fit) == 8
    with pytest.raises(FileNotFoundError):
        load_frozen_hst_mask(no_fit[0])


def test_nii_mask_width_and_guard() -> None:
    z = 0.3
    lc = REST_WAVELENGTH['Ha'] * (1 + z)
    wave = np.arange(lc - 40, lc + 40, 1.25)
    m = contaminant_channel_mask(wave, lc, z, HALPHA_CONTAMINANTS)
    assert m.any() and not m[np.abs(wave - lc) < lc * 350 / 299792.458].any()
    with pytest.raises(ValueError):
        contaminant_channel_mask(wave, lc, z, HALPHA_CONTAMINANTS, mask_halfwidth_kms=600.0)
