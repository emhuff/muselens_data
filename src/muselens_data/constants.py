"""Fixed numbers and service endpoints used to build the MUSElens data set.

Every value here is copied from the MUSElens production code (Phase 1 tag
``poolpsf``, 2026-09-21 onward). Changing one breaks the hash check against
the frozen manifest.
"""
from __future__ import annotations

# speed of light [km/s]
C_KMS: float = 299792.458

# --- emission lines (rest, Angstrom) ---
HA_REST: float = 6562.8
OII_BLUE: float = 3726.03
OII_RED: float = 3728.82
OII_CENTROID: float = (OII_BLUE + OII_RED) / 2.0
OIII_REST: float = 5006.84
OIII_4959_REST: float = 4958.91
NII_6548_REST: float = 6548.05
NII_6583_REST: float = 6583.45

# line fitted per target -> rest wavelength of the line window centre
REST_WAVELENGTH: dict[str, float] = {'Ha': HA_REST, 'OII': OII_CENTROID, 'OIII': OIII_REST}
# lines excluded from the continuum sidebands of the line window
SIDEBAND_CONTAMINANTS: dict[str, tuple[float, ...]] = {'Ha': (), 'OII': (), 'OIII': (OIII_4959_REST,)}
# lines masked (channel mask) inside the Halpha fit window
HALPHA_CONTAMINANTS: tuple[float, ...] = (NII_6548_REST, NII_6583_REST)

# line window used by Phase 1 [Angstrom]: half-width and sideband width
LINE_HALF_WINDOW: float = 40.0
CONTINUUM_SIDEBAND: float = 50.0

# --- HST ---
# CANDELS GOODS-S ACS/WFC F814W v1.0 mosaic, drizzled to 0.03"/pix, pixfrac 0.8
HST_PIXEL_SCALE: float = 0.03
HST_NATIVE_PIXEL_SCALE: float = 0.05
CANDELS_PIXFRAC: float = 0.8
HLA_FITSCUT_URL: str = ('https://hla.stsci.edu/cgi-bin/fitscut.cgi'
                        '?red={dataset}&RA={ra}&Dec={dec}&size={size}&format=fits&apix={pixel_scale}')
CANDELS_F814W_SCI: str = 'hlsp_candels_hst_acs_gs-tot_f814w_v1.0_drz'
CANDELS_F814W_WHT: str = 'hlsp_candels_hst_acs_gs-tot_f814w_v1.0_wht'

# --- MUSE ---
MUSE_PIXEL_SCALE: float = 0.2
MUSEWIDE_CUTOUT_URL: str = ('https://musewide.aip.de/cutout/api/datacubes/'
                            '?RA={ra:.6f}&A={box}&DEC={dec:.6f}&B={box}&L0={l0:.0f}&L1={l1:.0f}')
AMUSED_SOURCE_URL: str = 'https://amused.univ-lyon1.fr/data/UDF/HUDF/sources/{name}'
HUDF_UID_OFFSET: int = 900_000_000
