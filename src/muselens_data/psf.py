"""PSF and LSF of the MUSElens data, as used by the production fits.

HST F814W
    The production image model uses a circular Gaussian optical PSF of FWHM
    0.09", convolved with the CANDELS drizzle kernel (a square top hat of
    side pixfrac x native pixel = 0.8 x 0.05" = 0.04"), and rendered with
    GalSim ``drawImage`` in its default mode, which also integrates over the
    0.03" output pixel. This is a simplification of the true ACS PSF.

MUSE-Wide (fields candels-cdfs-NN)
    The cutouts carry no PSF keywords. Production uses ``psf_model =
    'measured'``: the Moffat column of Urrutia et al. 2019, Table 2 (FWHM
    linear in wavelength about 7050 A, beta constant per field) with the
    FWHM multiplied by a per-field scale measured by MUSElens on continuum
    images (``MUSE_PSF_FIELD_SCALE``; fields with fewer than 4 measurements
    carry the survey-wide value 0.937). The Gaussian column is kept for
    reference; it is not what production uses.

MUSE HUDF (fields hudf-mosaic, hudf-udf10, hudf-mxdf)
    The per-source MPDAF FSF (FSFMODE 2: FWHM and beta polynomials in reduced
    wavelength) is copied into the primary header of each converted cube and
    takes precedence over any table.

LSF (all cubes)
    Gaussian, FWHM = a lambda^2 + b lambda + c (Bacon et al. 2017, UDF-10
    coefficients, ``lsf_model = 'bacon2017_udf10'``), sigma = FWHM / 2.355.
    Fitted to sampled pixels, so no extra channel integration is applied.

Tables below are copied verbatim from ``muselens/instrument.py``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from astropy.io import fits

__all__ = [
    'HST_PSF_FWHM_ARCSEC', 'HST_DRIZZLE_KERNEL_ARCSEC', 'MUSE_LAMBDA_REF', 'BACON2017_LSF_FWHM_COEFFS',
    'MUSE_PSF_PARAMS', 'MUSE_PSF_MOFFAT_PARAMS', 'MUSE_PSF_FIELD_SCALE', 'MoffatFSF', 'read_moffat_fsf',
    'musewide_psf', 'lsf_sigma_angstrom', 'muse_psf_for_cube', 'PSFModel',
]

# HST optical PSF (Gaussian FWHM) and drizzle kernel (square, side) [arcsec]
HST_PSF_FWHM_ARCSEC: float = 0.09
HST_DRIZZLE_KERNEL_ARCSEC: float = 0.8 * 0.05

# reference wavelength for PSF parameterization [Angstrom]
MUSE_LAMBDA_REF: float = 7050.0

# empirical LSF FWHM [A] = a lambda^2 + b lambda + c (Bacon+2017 Eqs. 7-8;
# DR2 MXDF uses the udf10 form, Bacon+2023 Sect. 4.2.2)
BACON2017_LSF_FWHM_COEFFS: dict[str, tuple[float, float, float]] = {
    'bacon2017_udf10': (5.866e-8, -9.187e-4, 6.040),
    'bacon2017_mosaic': (5.835e-8, -9.080e-4, 5.983),
}

# Gaussian PSF FWHM at reference wavelength, per field (Urrutia+2019, Table 2)
# p0 [arcsec], p1 [arcsec/Angstrom]
MUSE_PSF_PARAMS: dict[str, tuple[float, float]] = {
    'candels-cdfs-01': (0.836, -4.429e-5),
    'candels-cdfs-02': (0.940, -3.182e-5),
    'candels-cdfs-03': (0.944, -4.460e-5),
    'candels-cdfs-04': (0.747, -4.218e-5),
    'candels-cdfs-05': (1.025, -3.003e-5),
    'candels-cdfs-06': (0.835, -4.331e-5),
    'candels-cdfs-07': (0.935, -3.966e-5),
    'candels-cdfs-08': (0.990, -5.007e-5),
    'candels-cdfs-09': (0.832, -8.069e-5),
    'candels-cdfs-10': (0.889, -3.050e-5),
    'candels-cdfs-11': (0.988, -3.770e-5),
    'candels-cdfs-12': (1.019, -4.122e-5),
    'candels-cdfs-13': (1.063, -5.284e-5),
    'candels-cdfs-14': (0.884, -4.843e-5),
    'candels-cdfs-15': (0.702, -4.441e-5),
    'candels-cdfs-16': (0.858, -3.784e-5),
    'candels-cdfs-17': (0.780, -3.534e-5),
    'candels-cdfs-18': (0.929, -3.478e-5),
    'candels-cdfs-19': (0.814, -3.524e-5),
    'candels-cdfs-20': (0.712, -5.196e-5),
    'candels-cdfs-21': (0.835, -4.255e-5),
    'candels-cdfs-22': (0.787, -3.252e-5),
    'candels-cdfs-23': (0.777, -3.018e-5),
    'candels-cdfs-24': (0.728, -4.232e-5),
    'candels-cdfs-25': (0.830, -4.310e-5),
    'candels-cdfs-26': (2.002, -2.630e-5),
    'candels-cdfs-28': (0.963, -4.439e-5),
    'candels-cdfs-29': (1.060, -3.550e-5),
    'candels-cdfs-30': (1.165, -5.690e-5),
    'candels-cdfs-31': (0.969, -4.450e-5),
    'candels-cdfs-32': (1.275, -5.980e-5),
    'candels-cdfs-33': (1.292, -4.900e-5),
    'candels-cdfs-34': (1.286, -8.770e-5),
    'candels-cdfs-35': (0.736, -2.840e-5),
    'candels-cdfs-36': (0.833, -3.860e-5),
    'candels-cdfs-37': (0.884, -2.920e-5),
    'candels-cdfs-39': (0.753, -3.420e-5),
    'candels-cdfs-40': (1.336, -6.290e-5),
    'candels-cdfs-41': (1.097, -5.160e-5),
    'candels-cdfs-42': (1.380, -5.840e-5),
    'candels-cdfs-43': (1.217, -3.640e-5),
    'candels-cdfs-44': (0.971, -6.280e-5),
    'candels-cdfs-45': (0.893, -3.140e-5),
    'candels-cdfs-46': (0.859, -4.610e-5),
}


# Moffat PSF at the reference wavelength, per field (Urrutia+2019, Table 2)
# p0 [arcsec], p1 [arcsec/Angstrom], beta (constant with wavelength)
MUSE_PSF_MOFFAT_PARAMS: dict[str, tuple[float, float, float]] = {
    'candels-cdfs-01': (0.744, -3.528e-05, 2.800),
    'candels-cdfs-02': (0.851, -4.060e-05, 2.983),
    'candels-cdfs-03': (0.809, -4.060e-05, 2.859),
    'candels-cdfs-04': (0.649, -2.626e-05, 2.800),
    'candels-cdfs-05': (1.148, -4.693e-05, 2.800),
    'candels-cdfs-06': (0.734, -3.044e-05, 2.800),
    'candels-cdfs-07': (0.871, -3.663e-05, 2.800),
    'candels-cdfs-08': (0.973, -5.670e-05, 2.434),
    'candels-cdfs-09': (0.726, -2.180e-05, 2.857),
    'candels-cdfs-10': (0.794, -5.150e-05, 2.622),
    'candels-cdfs-11': (0.934, -5.410e-05, 2.815),
    'candels-cdfs-12': (1.096, -5.564e-05, 2.800),
    'candels-cdfs-13': (1.166, -5.332e-05, 2.800),
    'candels-cdfs-14': (0.817, -5.610e-05, 2.896),
    'candels-cdfs-15': (0.735, -4.840e-05, 2.917),
    'candels-cdfs-16': (0.681, -4.070e-05, 2.657),
    'candels-cdfs-17': (0.644, -4.690e-05, 2.245),
    'candels-cdfs-18': (0.804, -3.505e-05, 2.800),
    'candels-cdfs-19': (0.676, -2.603e-05, 2.800),
    'candels-cdfs-20': (0.670, -5.006e-05, 2.800),
    'candels-cdfs-21': (0.598, -2.930e-05, 3.029),
    'candels-cdfs-22': (0.725, -5.710e-05, 3.078),
    'candels-cdfs-23': (0.720, -3.818e-05, 2.800),
    'candels-cdfs-24': (0.634, -3.190e-05, 2.800),
    'candels-cdfs-25': (0.696, -3.590e-05, 3.108),
    'candels-cdfs-26': (1.672, -6.483e-05, 2.800),
    'candels-cdfs-28': (0.830, -2.667e-05, 2.800),
    'candels-cdfs-29': (0.955, -3.190e-05, 3.266),
    'candels-cdfs-30': (1.021, -4.370e-05, 3.681),
    'candels-cdfs-31': (0.794, -4.600e-05, 2.593),
    'candels-cdfs-32': (0.968, -2.240e-06, 2.800),
    'candels-cdfs-33': (1.137, -5.140e-05, 2.800),
    'candels-cdfs-34': (0.941, -4.821e-05, 2.800),
    'candels-cdfs-35': (0.656, -3.052e-05, 2.800),
    'candels-cdfs-36': (0.758, -4.100e-05, 2.293),
    'candels-cdfs-37': (0.717, -4.109e-05, 2.800),
    'candels-cdfs-39': (0.638, -3.430e-05, 2.333),
    'candels-cdfs-40': (1.124, -4.575e-05, 2.800),
    'candels-cdfs-41': (0.880, -2.270e-05, 2.696),
    'candels-cdfs-42': (1.190, -6.794e-05, 2.800),
    'candels-cdfs-43': (1.010, -1.526e-05, 2.800),
    'candels-cdfs-44': (0.849, -3.528e-05, 2.800),
    'candels-cdfs-45': (0.756, -3.140e-05, 3.119),
    'candels-cdfs-46': (0.707, -2.990e-05, 2.279),
}


# Measured FWHM scale of the Moffat column, per field (experiments/psf_validation, 2026-09-21):
# continuum images of the pool galaxies and HST-selected point sources fitted with
# the HST F814W image convolved with a Moffat at the tabulated beta; fields with
# fewer than 4 measurements carry the survey-wide value. (scale, error, n_measurements)
MUSE_PSF_FIELD_SCALE: dict[str, tuple[float, float, int]] = {
    '_global': (0.937, 0.014, 176),
    'candels-cdfs-01': (0.937, 0.014, 2),
    'candels-cdfs-02': (0.911, 0.040, 12),
    'candels-cdfs-03': (1.079, 0.116, 4),
    'candels-cdfs-04': (1.126, 0.145, 4),
    'candels-cdfs-05': (0.878, 0.067, 11),
    'candels-cdfs-06': (0.957, 0.061, 7),
    'candels-cdfs-07': (0.856, 0.037, 4),
    'candels-cdfs-08': (0.824, 0.072, 6),
    'candels-cdfs-09': (1.019, 0.056, 5),
    'candels-cdfs-10': (0.937, 0.014, 1),
    'candels-cdfs-11': (0.937, 0.014, 0),
    'candels-cdfs-12': (0.937, 0.014, 1),
    'candels-cdfs-13': (0.937, 0.014, 2),
    'candels-cdfs-14': (0.855, 0.037, 9),
    'candels-cdfs-15': (0.937, 0.014, 2),
    'candels-cdfs-16': (1.181, 0.174, 5),
    'candels-cdfs-17': (1.014, 0.046, 5),
    'candels-cdfs-18': (0.937, 0.014, 3),
    'candels-cdfs-19': (0.937, 0.014, 3),
    'candels-cdfs-20': (0.937, 0.014, 1),
    'candels-cdfs-21': (0.937, 0.014, 2),
    'candels-cdfs-22': (0.863, 0.052, 4),
    'candels-cdfs-23': (0.961, 0.032, 11),
    'candels-cdfs-24': (0.930, 0.030, 5),
    'candels-cdfs-25': (0.937, 0.014, 2),
    'candels-cdfs-26': (0.937, 0.014, 0),
    'candels-cdfs-28': (0.937, 0.014, 2),
    'candels-cdfs-29': (0.937, 0.014, 1),
    'candels-cdfs-30': (0.937, 0.014, 2),
    'candels-cdfs-31': (0.883, 0.058, 6),
    'candels-cdfs-32': (0.958, 0.064, 4),
    'candels-cdfs-33': (0.937, 0.014, 3),
    'candels-cdfs-34': (0.937, 0.014, 1),
    'candels-cdfs-35': (0.937, 0.014, 2),
    'candels-cdfs-36': (0.937, 0.014, 2),
    'candels-cdfs-37': (0.909, 0.054, 7),
    'candels-cdfs-39': (0.848, 0.038, 6),
    'candels-cdfs-40': (0.937, 0.014, 2),
    'candels-cdfs-41': (0.937, 0.014, 2),
    'candels-cdfs-42': (0.794, 0.051, 5),
    'candels-cdfs-43': (0.945, 0.046, 4),
    'candels-cdfs-44': (1.004, 0.112, 4),
    'candels-cdfs-45': (1.076, 0.108, 5),
    'candels-cdfs-46': (0.892, 0.043, 7),
}


@dataclass(frozen=True)
class PSFModel:
    """A circular PSF at one wavelength. ``profile`` is 'moffat' or 'gaussian'; beta is None for a Gaussian."""
    profile: str
    fwhm_arcsec: float
    beta: float | None
    source: str


def musewide_psf(field_name: str, wavelength: float, model: str = 'measured') -> PSFModel:
    """MUSE-Wide PSF at ``wavelength`` [A]. ``model``: 'measured' (production), 'moffat' or 'gaussian'
    (the two Urrutia et al. 2019 Table 2 columns)."""
    if model == 'gaussian':
        table: dict = MUSE_PSF_PARAMS
    elif model in ('moffat', 'measured'):
        table = MUSE_PSF_MOFFAT_PARAMS
    else:
        raise ValueError(f"unknown MUSE PSF model {model!r}; use 'measured', 'moffat' or 'gaussian'")
    if field_name not in table:
        raise KeyError(f'unknown MUSE-Wide field {field_name!r}')
    p0, p1 = table[field_name][:2]
    fwhm = p0 + p1 * (wavelength - MUSE_LAMBDA_REF)
    if model == 'measured':
        fwhm *= MUSE_PSF_FIELD_SCALE[field_name][0]
    if fwhm <= 0:
        raise ValueError(f'non-positive PSF FWHM {fwhm:.3f}" for {field_name} at {wavelength:.1f} A')
    if model == 'gaussian':
        return PSFModel('gaussian', float(fwhm), None, 'Urrutia+2019 Table 2, Gaussian column')
    beta = float(table[field_name][2])
    src = 'Urrutia+2019 Table 2, Moffat column' + (' x MUSElens per-field scale' if model == 'measured' else '')
    return PSFModel('moffat', float(fwhm), beta, src)


class MoffatFSF:
    """Wavelength-dependent circular Moffat FSF (MPDAF FSFMODE=2).

    FWHM and beta are polynomials in the reduced wavelength (lbda - lb1) / (lb2 - lb1) - 0.5, numpy.polyval
    coefficient order (highest degree first).
    """

    def __init__(self, lb1: float, lb2: float, fwhm_pol: np.ndarray, beta_pol: np.ndarray) -> None:
        if lb2 <= lb1:
            raise ValueError(f'FSF reference wavelengths out of order: {lb1}, {lb2}')
        self.lb1, self.lb2 = float(lb1), float(lb2)
        self.fwhm_pol = np.asarray(fwhm_pol, dtype=np.float64)
        self.beta_pol = np.asarray(beta_pol, dtype=np.float64)

    def _reduced(self, wavelength: float) -> float:
        return (wavelength - self.lb1) / (self.lb2 - self.lb1) - 0.5

    def at(self, wavelength: float) -> PSFModel:
        fwhm = float(np.polyval(self.fwhm_pol, self._reduced(wavelength)))
        beta = float(np.polyval(self.beta_pol, self._reduced(wavelength)))
        if fwhm <= 0:
            raise ValueError(f'non-positive FSF FWHM {fwhm:.3f}" at {wavelength:.1f} A')
        if beta <= 1.0:
            raise ValueError(f'FSF Moffat beta {beta:.3f} <= 1 at {wavelength:.1f} A')
        return PSFModel('moffat', fwhm, beta, 'MPDAF FSF keywords (HUDF DR2)')


def read_moffat_fsf(header: fits.Header) -> MoffatFSF | None:
    """Parse MPDAF FSF keywords from a primary header; None if absent. Raises for FSF modes other than 2."""
    if 'FSFMODE' not in header:
        return None
    if header['FSFMODE'] != 2:
        raise ValueError(f'unsupported FSFMODE={header["FSFMODE"]}; only mode 2 is implemented')
    ids = sorted({m.group(1) for k in header if (m := re.fullmatch(r'FSF(\d{2})FNC', k))})
    if len(ids) != 1:
        raise ValueError(f'expected one FSF coefficient block, found {ids}')
    ff = ids[0]
    fwhm_pol = [header[f'FSF{ff}F{i:02d}'] for i in range(header[f'FSF{ff}FNC'])]
    beta_pol = [header[f'FSF{ff}B{i:02d}'] for i in range(header[f'FSF{ff}BNC'])]
    return MoffatFSF(header['FSFLB1'], header['FSFLB2'], np.array(fwhm_pol), np.array(beta_pol))


def muse_psf_for_cube(primary_header: fits.Header, field_name: str, wavelength: float) -> PSFModel:
    """The PSF production uses for a cube: the header FSF if present (HUDF), else the measured MUSE-Wide model.

    Raises for a HUDF field without FSF keywords, as production does.
    """
    fsf = read_moffat_fsf(primary_header)
    if fsf is not None:
        return fsf.at(wavelength)
    if field_name.startswith('hudf'):
        raise ValueError(f'HUDF field {field_name} but the cube has no FSF keywords')
    return musewide_psf(field_name, wavelength, model='measured')


def lsf_sigma_angstrom(wavelength: float, model: str = 'bacon2017_udf10') -> float:
    """Gaussian LSF sigma [A] at ``wavelength`` [A] (production: 'bacon2017_udf10')."""
    if model not in BACON2017_LSF_FWHM_COEFFS:
        raise ValueError(f'unknown LSF model {model!r}; use one of {sorted(BACON2017_LSF_FWHM_COEFFS)}')
    a, b, c = BACON2017_LSF_FWHM_COEFFS[model]
    fwhm = a * wavelength ** 2 + b * wavelength + c
    if fwhm <= 0:
        raise ValueError(f'non-positive LSF FWHM {fwhm:.3f} A at {wavelength:.1f} A')
    return fwhm / 2.355
