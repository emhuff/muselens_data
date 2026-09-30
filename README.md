# muselens_data

Standalone access to the data used by the MUSElens kinematic lensing analysis. It has no dependency on MUSElens,
kl_pipe, JAX or GalSim; it needs numpy, scipy, astropy, requests and sep.

It downloads, from the public archives, with the exact requests MUSElens made:

| product | source | per target |
|---|---|---|
| HST ACS F814W science + weight cutouts | HLA fitscut, CANDELS GOODS-S v1.0 mosaic (`..._drz`, `..._wht`), 0.03"/pix | square, 5 x r_circ half-width (r_circ = van der Wel r_e sqrt(q)) |
| MUSE-Wide datacube (290 targets) | musewide.aip.de cutout API, box 6 | +/- 250 A about the line, or the full range, as MUSElens has it |
| MUSE HUDF datacube (76 targets) | AMUSED DR2 per-source MPDAF bundle, rewritten to the MUSE-Wide layout | 25 x 25 spaxels, full range, FSF keywords kept |
| catalogues | MUSE-Wide DR1 main + emission-line tables, van der Wel 2012, Santini 2015, Skelton 2014, HUDF DR2 main + lines | - |

and ships, inside the package (2.8 MB):

- `data/targets.json`: the 366-target table (MUSElens `pool.json` records, the archive requests, the production
  instrument values, and the flags `in_phase1` (358) and `production` (the 128 rotators of the production
  shear run)).
- `data/masks/<id>.npz`: the HST segmentation and good-pixel masks saved by the production fits (358).
- `data/manifest.json`: content hashes of every file MUSElens used.
- The PSF and LSF tables (`psf.py`).

## Install and fetch

    pip install -e .            # or: pip install -e '.[test]'
    muselens-data fetch /path/to/data                      # all 366 targets + catalogues, ~12 GB
    muselens-data fetch /path/to/data --subset production  # the 128 production rotators
    muselens-data fetch /path/to/data --ids 101002010 900000863

`fetch` downloads whatever is missing and then checks every file against the manifest. The exit status is
non-zero if any file differs. `muselens-data verify DIR` repeats the check on its own.

**Skipping the check.** `muselens-data fetch DIR --no-verify` downloads without comparing to the manifest. Use it
when an archive has legitimately updated a product and the new version is wanted. Data fetched this way are no
longer the data MUSElens used. If the HST pixels changed, the frozen masks may not fit them: regenerate the masks
(below) instead of loading the frozen ones. `verify` can still be run afterwards to see what changed. The library
functions (`fetch_*`, `load_*`) never verify on their own.

**What "identical" means.** For every FITS HDU, the data array (dtype, shape, bytes) and every header card except
DATE, CHECKSUM and DATASUM must match. The HLA service stamps DATE on each request, so HST files are never
byte-identical. MUSE-Wide cubes, converted HUDF cubes and most catalogues are. The VizieR TSV headers record the
query date, so those two are compared on their data lines.

## Use

```python
from muselens_data import (get_target, target_ids, load_hst, load_muse, load_frozen_hst_mask,
                           regenerate_hst_mask, hst_variance, prepare_line_window)

t = get_target('101002010')           # record: ra, dec, redshift, line, field, masses, morphology seed, ...
hst = load_hst('101002010', DATA)     # image [e/s], weight [(e/s)^-2], header, 2D WCS, pixel scale
mask = load_frozen_hst_mask('101002010')
var = hst_variance(hst, mask)         # calibrated variance as the production image likelihood used it (inf = masked)
muse = load_muse('101002010', DATA)   # cube [1e-20 erg/s/cm^2/A], EFF_STAT variance, wave [A], 2D WCS, PSF
lw = prepare_line_window('101002010', DATA)   # production line window: continuum-subtracted data, calibrated
                                              # variance, [NII] channel mask, PSF and LSF at the line,
                                              # MUSE-vs-HST grid rotation
```

## PSF, LSF, WCS

- **HST.** Production uses a circular Gaussian of FWHM 0.09", convolved with the CANDELS drizzle kernel (a square
  top hat, 0.04" side) and drawn with GalSim's default method (so the 0.03" pixel is integrated too). This is a
  simplification of the true ACS PSF.
- **MUSE-Wide.** Moffat, from the Urrutia et al. 2019 Table 2 Moffat column (FWHM linear in wavelength about
  7050 A, beta per field), times a per-field FWHM scale measured by MUSElens on continuum images (0.79-1.18; the
  survey-wide 0.937 where a field has fewer than 4 measurements). `psf.musewide_psf(field, wavelength)`.
- **HUDF.** The MPDAF FSF in the cube's primary header (Moffat, FWHM and beta polynomials in wavelength).
- **LSF.** Gaussian, FWHM from the Bacon et al. 2017 UDF-10 polynomial. `psf.lsf_sigma_angstrom(wavelength)`.
- **WCS.** Taken from the FITS headers. MUSE pixel scales are sqrt|det CD|, because the HUDF MOSAIC grid is rotated
  by about 42 deg relative to the HST grid (`LineWindow.grid_rotation_rad`). For the 358 production targets the
  rebuilt PSF FWHM, LSF sigma, pixel scale and grid rotation equal the values stored by the production fits
  (`tests/test_data.py`).

## Masks and noise

The steps are in `masks.py` (module docstring):

- **HST.** Two-pass SEP detection. The target is the segment at the cutout centre. A narrow-band veto merges into
  the target any neighbour segment that sits on line flux at the target's wavelength. The weight map is rescaled
  so its sigma matches the blank-pixel MAD. Neighbours and weight <= 0 pixels are masked.
- **MUSE.** A +/- 40 A line window, with the continuum (median of two 50 A sidebands) subtracted. Each channel's
  variance is rescaled so its sigma matches the MAD of the spaxels that no HST source touches. For Halpha, channels
  within +/- 250 km/s of [NII] are masked.

`regenerate_hst_mask` recomputes the HST masks from the fetched files. `muselens-data masks DIR --subset phase1`
compares all 358 with the frozen production masks; on the MUSElens files they agree pixel for pixel.

## Things to know

- The MUSE-Wide cubes do not share a wavelength range: 227 are +/- 250 A windows (clipped at the MUSE range edge where needed) and 63 are full range,
  because MUSElens fetched them at different times. The requests in `targets.json` reproduce each one.
- 8 targets have no production fit (`in_phase1` false) and therefore no frozen mask.
- Selection flags are informational; nothing here applies a selection.
- `tools/freeze_from_muselens.py` built the package data from the MUSElens working copy on 2026-09-30. It is
  kept as the record of how that was done.

## Tests

    pytest tests/test_offline.py                                  # no data needed
    MUSELENS_DATA_DIR=/path/to/data pytest tests/test_data.py     # after fetch

## References

Urrutia et al. 2019 (MUSE-Wide DR1); Bacon et al. 2017, 2023 (MUSE HUDF, LSF); Koekemoer et al. 2011, Grogin et
al. 2011 (CANDELS); van der Wel et al. 2012; Santini et al. 2015; Skelton et al. 2014.
