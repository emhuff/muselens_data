"""Masks and noise calibration of the MUSElens production (Phase 1, tag poolpsf) data preparation.

HST (per target)
    1. SEP detection on the F814W cutout, two passes (pass 1 uses the weight-map noise, pass 2 masks the pass-1
       sources and uses the background RMS); threshold 1.5 sigma, deblend_cont 0.01, 32 thresholds, min area 5.
    2. Target = the segment containing the cutout centre pixel. Every other segment is a neighbour.
    3. Narrow-band veto: a neighbour segment whose MUSE footprint sits on narrow-band line flux at the target's
       line wavelength is part of the target and is merged into it (``classify_segments_nb``).
    4. Variance = 1 / (weight x s), s a single factor that makes the weight-map sigma equal the MAD sigma of the
       blank (segmap == 0) pixels. Pixels with weight <= 0 and neighbour pixels get infinite variance.
       good_mask = finite variance.

MUSE (per target, derived at load time from the HST segmentation)
    5. Line window of +/- 40 A about the line, continuum = per-spaxel median of two 50 A sidebands (with
       [OIII]4959 excluded for OIII targets) subtracted.
    6. Blank spaxels = MUSE spaxels hit by no HST source pixel of the pass-2 SEP segmentation (no veto), found by
       projecting every source pixel centre through the two WCSs.
    7. Per-channel variance factor = (MAD sigma / median pipeline sigma)^2 over the blank spaxels.
    8. Halpha: channels within +/- 250 km/s of [NII]6548,6583 get infinite variance.
    9. Non-finite data or variance, or variance <= 0: data 0, variance infinite.

The frozen HST masks shipped in data/masks are the outputs of steps 1-4 as saved by the production run;
``regenerate_hst_mask`` recomputes them and ``tests/test_masks.py`` checks that the two agree exactly.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from importlib import resources
from typing import Any

import numpy as np
import sep
from astropy.wcs import WCS
from scipy import ndimage

from muselens_data.constants import C_KMS

__all__ = ['detect_sources', 'rescale_variance', 'classify_segments_nb', 'nb_sigma_from_image', 'VetoNotApplicable',
           'SegmentDecision', 'extract_line_window', 'narrowband_image', 'project_sources_to_muse',
           'rescale_variance_cube', 'contaminant_channel_mask', 'frame_rotation', 'load_frozen_hst_mask',
           'HSTMask']

# narrow-band veto constants (muselens/segment_veto.py)
DILATE_HST_PIX: int = 4
NB_DETECT_SIGMA: float = 3.0
MIN_TARGET_NB_SPAXELS: int = 10
NB_HALF_WIDTH_KMS: float = 500.0


# ----------------------------------------------------------------------------------------------------------------
# HST
# ----------------------------------------------------------------------------------------------------------------

def detect_sources(image: np.ndarray, weight: np.ndarray, detect_thresh: float = 1.5, deblend_cont: float = 0.01,
                   n_iter: int = 2, min_area: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Iterative SEP detection (step 1). Returns the segmentation map (0 = background) and the SEP catalogue."""
    if image.shape != weight.shape:
        raise ValueError(f'image shape {image.shape} != weight shape {weight.shape}')
    data = np.ascontiguousarray(image, dtype=np.float64)
    good = weight > 0
    err = np.full_like(data, np.inf)
    err[good] = 1.0 / np.sqrt(weight[good])
    source_mask = np.zeros(data.shape, dtype=bool)
    objects = segmap = None
    for i in range(n_iter):
        bkg = sep.Background(data, mask=source_mask, bw=32, bh=32)
        data_sub = data - bkg.back()
        detect_err = err if i == 0 else bkg.rms()
        objects, segmap = sep.extract(data_sub, thresh=detect_thresh, err=detect_err, deblend_cont=deblend_cont,
                                      deblend_nthresh=32, minarea=min_area, segmentation_map=True)
        source_mask = segmap > 0
    return segmap, objects


def rescale_variance(image: np.ndarray, weight: np.ndarray, segmap: np.ndarray) -> tuple[np.ndarray, float]:
    """Variance from the weight map with one global factor matching the blank-pixel MAD noise (step 4).

    Returns (variance, s) with variance = 1 / (weight s) where weight > 0, inf elsewhere. Raises with fewer than
    100 blank pixels.
    """
    blank = (segmap == 0) & (weight > 0) & np.isfinite(image)
    n_blank = int(blank.sum())
    if n_blank < 100:
        raise ValueError(f'only {n_blank} blank pixels for the variance calibration (need >= 100)')
    px = image[blank]
    sigma_empirical = 1.4826 * np.median(np.abs(px - np.median(px)))
    sigma_weightmap = np.median(1.0 / np.sqrt(weight[blank]))
    s = (sigma_empirical / sigma_weightmap) ** 2
    variance = np.full_like(weight, np.inf)
    good = weight > 0
    variance[good] = 1.0 / (weight[good] * s)
    return variance, float(s)


class VetoNotApplicable(Exception):
    """The target's narrow-band detection is too small for the veto to carry evidence."""


@dataclass
class SegmentDecision:
    label: int
    n_hst_pix: int
    n_muse_spaxels: int
    overlap_frac: float
    mean_nb_snr: float
    merge: bool


def _muse_to_hst_index(hst_wcs: WCS, muse_wcs: WCS, shape_muse: tuple[int, int],
                       shape_hst: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = shape_muse
    yy, xx = np.mgrid[0:ny, 0:nx]
    ra, dec = muse_wcs.all_pix2world(xx.ravel(), yy.ravel(), 0)
    hx, hy = hst_wcs.all_world2pix(ra, dec, 0)
    hx = np.round(hx).astype(int).reshape(ny, nx)
    hy = np.round(hy).astype(int).reshape(ny, nx)
    inside = (hx >= 0) & (hx < shape_hst[1]) & (hy >= 0) & (hy < shape_hst[0])
    return hx, hy, inside


def classify_segments_nb(segmap: np.ndarray, target_label: int, hst_wcs: WCS, muse_wcs: WCS, narrowband: np.ndarray,
                         nb_sigma: float, min_overlap_frac: float = 0.5, min_int_snr: float = 5.0,
                         min_overlap_frac_snr: float = 0.2) -> list[SegmentDecision]:
    """Narrow-band veto (step 3). A neighbour segment is merged into the target when the fraction of its dilated
    MUSE footprint above 3 sigma in the narrow-band image is >= 0.5, or when its summed narrow-band S/N is >= 5 and
    that fraction is >= 0.2. Raises VetoNotApplicable when the target itself has < 10 detected spaxels."""
    if nb_sigma <= 0 or not np.isfinite(nb_sigma):
        raise ValueError(f'invalid nb_sigma {nb_sigma}')
    ny, nx = narrowband.shape
    hx, hy, inside = _muse_to_hst_index(hst_wcs, muse_wcs, (ny, nx), segmap.shape)
    nb_mask = narrowband > NB_DETECT_SIGMA * nb_sigma
    nb_snr = narrowband / nb_sigma

    def footprint(seg: np.ndarray) -> np.ndarray:
        seg_dil = ndimage.binary_dilation(seg, iterations=DILATE_HST_PIX)
        foot = np.zeros((ny, nx), dtype=bool)
        foot[inside] = seg_dil[hy[inside], hx[inside]]
        return foot

    n_target_nb = int((nb_mask & footprint(segmap == target_label)).sum())
    if n_target_nb < MIN_TARGET_NB_SPAXELS:
        raise VetoNotApplicable(f'target NB detection has {n_target_nb} spaxels under the target segment '
                                f'(< {MIN_TARGET_NB_SPAXELS}); the veto carries no evidence')
    decisions = []
    for lab in sorted(set(np.unique(segmap[segmap > 0])) - {target_label}):
        seg = segmap == lab
        foot = footprint(seg)
        n_sp = int(foot.sum())
        if n_sp == 0:
            decisions.append(SegmentDecision(int(lab), int(seg.sum()), 0, 0.0, 0.0, False))
            continue
        frac = float(nb_mask[foot].mean())
        mean_snr = float(nb_snr[foot].mean())
        int_snr = mean_snr * np.sqrt(n_sp)
        merge = bool((frac >= min_overlap_frac) or (int_snr >= min_int_snr and frac >= min_overlap_frac_snr))
        decisions.append(SegmentDecision(int(lab), int(seg.sum()), n_sp, frac, mean_snr, merge))
    return decisions


def nb_sigma_from_image(narrowband: np.ndarray) -> float:
    """MAD noise of a narrow-band image."""
    v = narrowband[np.isfinite(narrowband)]
    if v.size < 100:
        raise ValueError('too few finite narrow-band pixels for a noise estimate')
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


@dataclass
class HSTMask:
    segmap: np.ndarray        # int, 0 = background; merged neighbours carry the target label
    good_mask: np.ndarray     # bool, True where the production image likelihood used the pixel
    target_label: int
    veto: dict[str, Any]

    def neighbour_mask(self) -> np.ndarray:
        return (self.segmap > 0) & (self.segmap != self.target_label)


def load_frozen_hst_mask(galaxy_id: str) -> HSTMask:
    """The HST mask saved by the production Phase 1 fit (358 targets). Raises FileNotFoundError otherwise."""
    ref = resources.files('muselens_data').joinpath(f'data/masks/{galaxy_id}.npz')
    if not ref.is_file():
        raise FileNotFoundError(f'no frozen HST mask for {galaxy_id} (only targets with a production Phase 1 fit)')
    with ref.open('rb') as f, np.load(f) as m:
        return HSTMask(segmap=m['segmap'].astype(np.int32), good_mask=m['good_mask'].astype(bool),
                       target_label=int(m['target_label']), veto={'source': 'frozen'})


# ----------------------------------------------------------------------------------------------------------------
# MUSE
# ----------------------------------------------------------------------------------------------------------------

def extract_line_window(data_cube: np.ndarray, variance_cube: np.ndarray, wave: np.ndarray, line_center: float,
                        redshift: float, wave_window: float = 40.0, continuum_window: float = 50.0,
                        exclude_rest_wavelengths: tuple[float, ...] = (), exclude_halfwidth_kms: float = 600.0
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Continuum-subtracted line window (step 5; production ``continuum_source='sidebands'``).

    Returns (data_sub, var_sub, wave_mask, continuum). Spaxels with fewer than 5 finite sideband channels get a
    NaN continuum (masked downstream).
    """
    line_mask = (wave >= line_center - wave_window) & (wave <= line_center + wave_window)
    if not line_mask.any():
        raise ValueError(f'no channels in the line window around {line_center:.1f} A '
                         f'(cube {wave[0]:.1f}-{wave[-1]:.1f} A)')
    blue = (wave >= line_center - wave_window - continuum_window) & (wave < line_center - wave_window)
    red = (wave > line_center + wave_window) & (wave <= line_center + wave_window + continuum_window)
    for rest in exclude_rest_wavelengths:
        lam = rest * (1.0 + redshift)
        near = np.abs(wave - lam) < lam * exclude_halfwidth_kms / C_KMS
        if np.any(near & line_mask):
            raise ValueError(f'excluded line at {lam:.1f} A falls inside the fit window around {line_center:.1f} A')
        blue &= ~near
        red &= ~near
    if exclude_rest_wavelengths and (blue.sum() < 5 or red.sum() < 5):
        raise ValueError(f'sideband exclusion leaves {blue.sum()} blue / {red.sum()} red continuum channels')
    cont_mask = blue | red
    if not cont_mask.any():
        raise ValueError('no continuum channels in the sidebands')
    cont_cube = data_cube[cont_mask]
    n_fin = np.isfinite(cont_cube).sum(axis=0)
    # all-NaN spaxels (HUDF NaN channels) are expected; they get a NaN continuum and are masked downstream
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
        continuum = np.nanmedian(np.where(np.isfinite(cont_cube), cont_cube, np.nan), axis=0)
    continuum = np.where(n_fin >= 5, continuum, np.nan)
    return data_cube[line_mask] - continuum[None], variance_cube[line_mask], line_mask, continuum


def narrowband_image(data_sub: np.ndarray, sub_wave: np.ndarray, line_center: float) -> np.ndarray:
    """Sum of the continuum-subtracted channels within +/- 500 km/s of the line (the veto's narrow band)."""
    near = np.abs(C_KMS * (sub_wave - line_center) / line_center) < NB_HALF_WIDTH_KMS
    if near.sum() < 3:
        raise ValueError('narrow-band window has fewer than 3 channels')
    return np.nansum(data_sub[near], axis=0)


def project_sources_to_muse(source_mask_hst: np.ndarray, hst_wcs: WCS, muse_wcs: WCS,
                            shape_muse: tuple[int, int]) -> np.ndarray:
    """MUSE spaxels containing the centre of at least one HST source pixel (step 6)."""
    ny, nx = shape_muse
    out = np.zeros((ny, nx), dtype=bool)
    hy, hx = np.where(source_mask_hst)
    if len(hx) == 0:
        return out
    ra, dec = hst_wcs.all_pix2world(hx, hy, 0)
    mx, my = muse_wcs.all_world2pix(ra, dec, 0)
    mx = np.round(mx).astype(int)
    my = np.round(my).astype(int)
    ok = (mx >= 0) & (mx < nx) & (my >= 0) & (my < ny)
    out[my[ok], mx[ok]] = True
    return out


def rescale_variance_cube(data_sub: np.ndarray, var_sub: np.ndarray, blank_mask: np.ndarray,
                          min_blank: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel variance factor from the blank spaxels (step 7). Channels with fewer than ``min_blank`` usable
    blank spaxels take the median factor. Returns (rescaled variance, factors)."""
    n_blank = int(blank_mask.sum())
    if n_blank < min_blank:
        raise ValueError(f'only {n_blank} blank spaxels for the variance calibration (need >= {min_blank})')
    factors = np.full(data_sub.shape[0], np.nan)
    for i in range(data_sub.shape[0]):
        bf = data_sub[i, blank_mask]
        bv = var_sub[i, blank_mask]
        good = np.isfinite(bf) & np.isfinite(bv) & (bv > 0)
        if good.sum() < min_blank:
            continue
        bf, bv = bf[good], bv[good]
        sig_emp = 1.4826 * np.median(np.abs(bf - np.median(bf)))
        sig_pipe = np.median(np.sqrt(bv))
        if sig_emp > 0 and sig_pipe > 0:
            factors[i] = (sig_emp / sig_pipe) ** 2
    ok = np.isfinite(factors)
    if not ok.any():
        raise ValueError('no channel has a usable variance factor')
    factors[~ok] = float(np.median(factors[ok]))
    return var_sub * factors[:, None, None], factors


def contaminant_channel_mask(sub_wave: np.ndarray, line_center: float, redshift: float,
                             contaminant_rest_wavelengths: tuple[float, ...], mask_halfwidth_kms: float = 250.0,
                             primary_excursion_kms: float = 300.0, guard_kms: float = 50.0) -> np.ndarray:
    """Channels within +/- ``mask_halfwidth_kms`` of each contaminant line (step 8). Defined from the redshift, not
    the data. Raises if a mask would come within primary_excursion + guard of the primary line."""
    mask = np.zeros(sub_wave.shape, dtype=bool)
    for rest in contaminant_rest_wavelengths:
        offset = (rest * (1.0 + redshift) / line_center - 1.0) * C_KMS
        lo, hi = offset - mask_halfwidth_kms, offset + mask_halfwidth_kms
        inner = lo if offset > 0 else hi
        if abs(inner) < primary_excursion_kms + guard_kms:
            raise ValueError(f'contaminant {rest:.2f} A mask edge {inner:+.0f} km/s intrudes on the primary line')
        mask |= (sub_wave >= line_center * (1 + lo / C_KMS)) & (sub_wave <= line_center * (1 + hi / C_KMS))
    return mask


def frame_rotation(ref_wcs: WCS, muse_wcs: WCS, ra: float, dec: float) -> float:
    """Rotation [rad] of the MUSE pixel grid relative to the HST pixel grid at (ra, dec). Zero to 1e-3 for
    MUSE-Wide; about 42 deg for the HUDF MOSAIC cubes. Raises if the grids differ by more than rotation and scale."""
    eps = 1e-4

    def jac(w: WCS) -> np.ndarray:
        x0, y0 = w.all_world2pix(ra, dec, 0)
        xe, ye = w.all_world2pix(ra + eps / np.cos(np.radians(dec)), dec, 0)
        xn, yn = w.all_world2pix(ra, dec + eps, 0)
        return np.array([[xe - x0, xn - x0], [ye - y0, yn - y0]], dtype=float) / eps

    m = jac(muse_wcs) @ np.linalg.inv(jac(ref_wcs))
    r = m / np.sqrt(abs(np.linalg.det(m)))
    if np.linalg.det(r) < 0 or not np.allclose(r.T @ r, np.eye(2), atol=2e-3):
        raise ValueError(f'MUSE and HST grids differ by more than a rotation and scale: {m}')
    return float(np.arctan2(r[1, 0], r[0, 0]))


def decisions_to_json(decisions: list[SegmentDecision]) -> list[dict]:
    return [asdict(d) for d in decisions]
