"""Load the fetched files into arrays, with WCS, PSF, masks and calibrated noise as production used them.

Units: HST image in electrons/s (CANDELS drizzled mosaic), weight = inverse variance in (electrons/s)^-2 before
calibration. MUSE flux in 1e-20 erg/s/cm^2/A per spaxel, variance in its square, wavelength in Angstrom
from the cube header WCS. Pixel scales in arcsec.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from muselens_data.constants import (CONTINUUM_SIDEBAND, HALPHA_CONTAMINANTS, LINE_HALF_WINDOW, MUSE_PIXEL_SCALE,
                                     REST_WAVELENGTH, SIDEBAND_CONTAMINANTS)
from muselens_data.fetch import hst_paths, muse_path
from muselens_data.masks import (HSTMask, VetoNotApplicable, classify_segments_nb, contaminant_channel_mask,
                                 decisions_to_json, detect_sources, extract_line_window, frame_rotation,
                                 load_frozen_hst_mask, narrowband_image, nb_sigma_from_image, project_sources_to_muse,
                                 rescale_variance, rescale_variance_cube)
from muselens_data.psf import PSFModel, lsf_sigma_angstrom, muse_psf_for_cube
from muselens_data.targets import get_target

__all__ = ['HSTData', 'MUSEData', 'LineWindow', 'load_hst', 'load_muse', 'pixel_scale_from_header',
           'regenerate_hst_mask', 'hst_variance', 'prepare_line_window', 'line_center']


def pixel_scale_from_header(header: fits.Header) -> float:
    """Square pixel scale [arcsec] = sqrt|det CD| (rotation-invariant; |CD1_1| is wrong for rotated grids)."""
    if 'CD1_1' in header:
        cd = np.array([[header['CD1_1'], header.get('CD1_2', 0.0)], [header.get('CD2_1', 0.0), header['CD2_2']]])
    else:
        w = WCS(header, naxis=2)
        cd = w.wcs.get_pc() * w.wcs.get_cdelt()[:, None] if w.wcs.has_pc() else np.diag(w.wcs.get_cdelt())
    sx = float(np.hypot(cd[0, 0], cd[1, 0])) * 3600.0
    sy = float(np.hypot(cd[0, 1], cd[1, 1])) * 3600.0
    if not np.isfinite(sx) or sx <= 0 or abs(sx - sy) > 1e-3 * sx:
        raise ValueError(f'bad or non-square pixel scale {sx:.5f}" x {sy:.5f}"')
    return float(np.sqrt(abs(np.linalg.det(cd)))) * 3600.0


def line_center(target: dict[str, Any]) -> float:
    """Observed wavelength [A] of the fitted line: rest x (1 + z); [OII] uses the doublet centroid."""
    return REST_WAVELENGTH[target['line']] * (1.0 + float(target['redshift']))


@dataclass
class HSTData:
    galaxy_id: str
    image: np.ndarray
    weight: np.ndarray
    header: fits.Header
    wcs: WCS
    pixel_scale: float


@dataclass
class MUSEData:
    galaxy_id: str
    field: str
    data: np.ndarray            # (n_wave, ny, nx)
    variance: np.ndarray        # pipeline EFF_STAT (uncalibrated)
    wave: np.ndarray
    header: fits.Header         # DATA_DCBGC header
    primary_header: fits.Header
    wcs: WCS                    # spatial (2D)
    pixel_scale: float

    def psf_at(self, wavelength: float) -> PSFModel:
        return muse_psf_for_cube(self.primary_header, self.field, wavelength)


def load_hst(galaxy_id: str, data_dir: str | Path) -> HSTData:
    sci, wht = hst_paths(data_dir, galaxy_id)
    for p in (sci, wht):
        if not p.exists():
            raise FileNotFoundError(f'{p} missing; run fetch first')
    with fits.open(sci) as h:
        image = np.array(h[0].data, dtype=np.float64)
        header = h[0].header.copy()
    with fits.open(wht) as h:
        weight = np.array(h[0].data, dtype=np.float64)
    if image.shape != weight.shape:
        raise ValueError(f'{galaxy_id}: science {image.shape} != weight {weight.shape}')
    return HSTData(galaxy_id, image, weight, header, WCS(header, naxis=2), pixel_scale_from_header(header))


def load_muse(galaxy_id: str, data_dir: str | Path) -> MUSEData:
    t = get_target(galaxy_id)
    p = muse_path(data_dir, galaxy_id)
    if not p.exists():
        raise FileNotFoundError(f'{p} missing; run fetch first')
    with fits.open(p) as h:
        primary = h[0].header.copy()
        hdr = h[1].header.copy()
        data = np.array(h[1].data, dtype=np.float64)
        var = np.array(h[2].data, dtype=np.float64)
    if data.shape != var.shape:
        raise ValueError(f'{galaxy_id}: data {data.shape} != variance {var.shape}')
    n = hdr['NAXIS3']
    wave = hdr['CRVAL3'] + (np.arange(n) - (hdr.get('CRPIX3', 1.0) - 1.0)) * hdr.get('CD3_3', 1.25)
    scale = pixel_scale_from_header(hdr) if 'CD1_1' in hdr else MUSE_PIXEL_SCALE
    return MUSEData(galaxy_id, t['field'], data, var, np.asarray(wave, dtype=np.float64), hdr, primary,
                    WCS(hdr, naxis=2), scale)


def regenerate_hst_mask(galaxy_id: str, data_dir: str | Path, veto: bool | None = None) -> HSTMask:
    """Recompute the production HST mask (masks.py steps 1-4) from the fetched files.

    ``veto`` defaults to what the production fit did (``target['phase1']['nb_veto']`` is 'applied',
    'not_applicable' or 'off'); for a target without a production fit it defaults to True.
    """
    t = get_target(galaxy_id)
    mode = (t['phase1'] or {}).get('nb_veto', 'applied')
    if veto is None:
        veto = mode != 'off'
    hst = load_hst(galaxy_id, data_dir)
    segmap, _ = detect_sources(hst.image, hst.weight)
    variance, _ = rescale_variance(hst.image, hst.weight, segmap)
    cy, cx = segmap.shape[0] // 2, segmap.shape[1] // 2
    target_label = int(segmap[cy, cx])
    if target_label == 0:
        raise ValueError(f'{galaxy_id}: no HST segment at the cutout centre')
    variance[(segmap > 0) & (segmap != target_label)] = np.inf
    info: dict[str, Any] = {'applied': False}
    if veto:
        muse = load_muse(galaxy_id, data_dir)
        lc = line_center(t)
        data_sub, _, wmask, _ = extract_line_window(muse.data, muse.variance, muse.wave, lc, float(t['redshift']),
                                                    LINE_HALF_WINDOW, CONTINUUM_SIDEBAND,
                                                    SIDEBAND_CONTAMINANTS[t['line']])
        nb = narrowband_image(data_sub, muse.wave[wmask], lc)
        nb_sigma = nb_sigma_from_image(nb)
        try:
            decisions = classify_segments_nb(segmap, target_label, hst.wcs, muse.wcs, nb, nb_sigma)
        except VetoNotApplicable as e:
            info = {'applied': False, 'reason': str(e)}
        else:
            merged = [d.label for d in decisions if d.merge]
            for lab in merged:
                segmap[segmap == lab] = target_label
            variance, _ = rescale_variance(hst.image, hst.weight, segmap)
            variance[(segmap > 0) & (segmap != target_label)] = np.inf
            info = {'applied': True, 'merged_labels': merged, 'nb_sigma': nb_sigma,
                    'decisions': decisions_to_json(decisions)}
    return HSTMask(segmap=segmap.astype(np.int32), good_mask=np.isfinite(variance),
                   target_label=int(segmap[cy, cx]), veto=info)


def hst_variance(hst: HSTData, mask: HSTMask) -> np.ndarray:
    """Calibrated HST variance as the production likelihood used it: weight rescaled on the blank pixels of
    ``mask.segmap``, infinite outside ``mask.good_mask`` and on neighbour segments."""
    if mask.segmap.shape != hst.image.shape:
        raise ValueError(f'mask shape {mask.segmap.shape} != image {hst.image.shape}')
    var, _ = rescale_variance(hst.image, hst.weight, mask.segmap)
    var = np.where(mask.good_mask, var, np.inf)
    var[mask.neighbour_mask()] = np.inf
    return var


@dataclass
class LineWindow:
    galaxy_id: str
    line: str
    line_center: float           # observed [A]
    wave: np.ndarray             # (n_chan,)
    data: np.ndarray             # continuum-subtracted, (n_chan, ny, nx); 0 where masked
    variance: np.ndarray         # calibrated; inf where masked
    continuum: np.ndarray        # (ny, nx) subtracted sideband median
    blank_spaxels: np.ndarray    # (ny, nx) bool, used for the variance calibration
    variance_factors: np.ndarray # (n_chan,)
    contaminant_channels: np.ndarray  # (n_chan,) bool, [NII] for Halpha
    psf: PSFModel                # at line_center
    lsf_sigma_angstrom: float    # at line_center
    pixel_scale: float
    grid_rotation_rad: float     # MUSE grid relative to the HST grid


def prepare_line_window(galaxy_id: str, data_dir: str | Path) -> LineWindow:
    """The MUSE line window exactly as the production Phase 1 fit prepared it (masks.py steps 5-9)."""
    t = get_target(galaxy_id)
    hst = load_hst(galaxy_id, data_dir)
    muse = load_muse(galaxy_id, data_dir)
    lc = line_center(t)
    z = float(t['redshift'])
    data_sub, var_sub, wmask, continuum = extract_line_window(muse.data, muse.variance, muse.wave, lc, z,
                                                              LINE_HALF_WINDOW, CONTINUUM_SIDEBAND,
                                                              SIDEBAND_CONTAMINANTS[t['line']])
    sub_wave = muse.wave[wmask]
    seg_all, _ = detect_sources(hst.image, hst.weight)
    blank = ~project_sources_to_muse(seg_all > 0, hst.wcs, muse.wcs, data_sub.shape[1:])
    var_cal, factors = rescale_variance_cube(data_sub, var_sub, blank)
    bad = ~np.isfinite(data_sub) | ~np.isfinite(var_cal) | (var_cal <= 0)
    data_out = np.where(bad, 0.0, data_sub)
    var_out = np.where(bad, np.inf, var_cal)
    cmask = np.zeros(sub_wave.shape, dtype=bool)
    if t['line'] == 'Ha':
        cmask = contaminant_channel_mask(sub_wave, lc, z, HALPHA_CONTAMINANTS)
        var_out[cmask] = np.inf
    rot = frame_rotation(hst.wcs, muse.wcs, float(t['ra']), float(t['dec']))
    return LineWindow(galaxy_id, t['line'], lc, sub_wave, data_out, var_out, continuum, blank, factors, cmask,
                      muse.psf_at(lc), lsf_sigma_angstrom(lc), muse.pixel_scale, rot)
