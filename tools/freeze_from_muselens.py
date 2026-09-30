"""Build the package data from a MUSElens working copy (read-only access to it).

    python tools/freeze_from_muselens.py --muselens ../MUSElens --musewide ../MUSE-Wide

Writes src/muselens_data/data/{targets.json, manifest.json, catalogs.json, masks/<gid>.npz}. Run once, by the
MUSElens authors; the collaborator never needs it. Raises on any inconsistency between the pool table, the
Phase 1 records and the files on disk rather than guessing.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from datetime import date
from pathlib import Path

from astropy.io import fits

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'src'))
from muselens_data.constants import MUSEWIDE_CUTOUT_URL, REST_WAVELENGTH  # noqa: E402
from muselens_data.fingerprint import data_lines_sha256, file_sha256, fits_fingerprint  # noqa: E402

OUT = HERE / 'src' / 'muselens_data' / 'data'
PHASE1_TAG = 'poolpsf'
PRODUCTION_RUN = 'experiments/population/runs/shear_hann_poolpsf_th8'
MUSE_FETCH_HALFWIDTH = 250.0   # [A] scripts/fit_kinematic_cosi.py
MUSEWIDE_FULL = (4750.0, 9350.0)
TAP = 'https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync'
# MUSE-Wide cubes that were not requested at the pool position. The cutout grid depends on the requested position,
# so these need their own. Each override was found by refetching and is byte-identical to the MUSElens file.
MUSE_POSITION_OVERRIDES: dict[str, tuple[float, float, str]] = {
    '102030136': (53.05962049148014, -27.822399306831272,
                  'fetched 2026-03-27 by pre-git MUSE-Wide code at an unrecorded position; at the pool position the '
                  'server returns a grid shifted one spaxel (0.2 arcsec) in Dec. This position (the sky position of '
                  'pixel (30, 30) of the MUSElens cube) reproduces the file byte for byte (checked 2026-09-30).'),
}


def vizier_request(path: Path) -> str:
    for line in open(path):
        m = re.match(r'#INFO\trequest=(\S+)', line)
        if m:
            return m.group(1)
    raise ValueError(f'{path}: no VizieR request line')


def muse_request(t: dict, cube: Path) -> dict:
    gid = t['galaxy_id']
    h0 = fits.getheader(cube, 0)
    h1 = fits.getheader(cube, 1)
    if t['source'] == 'HUDF':
        name = f'DR2_{t["hudf_dataset"].upper()}_{int(t["hudf_id"]):06d}.fits'
        if h0.get('ORIGFILE') != name:
            raise ValueError(f'{gid}: cube ORIGFILE {h0.get("ORIGFILE")} != {name}')
        return {'service': 'amused', 'source_file': name, 'dataset': t['hudf_dataset'], 'hudf_id': int(t['hudf_id'])}
    n, crval = h1['NAXIS3'], h1['CRVAL3']
    if n > 3000:
        l0, l1 = MUSEWIDE_FULL
    else:
        lam = REST_WAVELENGTH[t['line']] * (1 + t['redshift'])
        l0 = float(f'{max(lam - MUSE_FETCH_HALFWIDTH, MUSEWIDE_FULL[0]):.0f}')
        l1 = float(f'{min(lam + MUSE_FETCH_HALFWIDTH, MUSEWIDE_FULL[1]):.0f}')
    ra, dec, note = t['ra'], t['dec'], None
    if gid in MUSE_POSITION_OVERRIDES:
        ra, dec, note = MUSE_POSITION_OVERRIDES[gid]
    wave_end = crval + (n - 1) * h1['CD3_3']
    if not (abs(crval - l0) <= 2.5 and abs(wave_end - l1) <= 2.5):  # server snaps to the native 1.25 A grid
        raise ValueError(f'{gid}: cube spans {crval:.2f}-{wave_end:.2f} A, request {l0}-{l1}')
    req = {'service': 'musewide_api', 'ra': ra, 'dec': dec, 'box_arcsec': 6, 'l0': l0, 'l1': l1,
           'url': MUSEWIDE_CUTOUT_URL.format(ra=ra, dec=dec, box=6, l0=l0, l1=l1)}
    if note is not None:
        req['note'] = note
    return req


def phase1_block(rec: dict, t: dict) -> dict:
    for k in ('ra', 'dec', 'redshift', 'line', 'field'):
        if rec[k] != t[k]:
            raise ValueError(f'{t["galaxy_id"]}: Phase 1 {k}={rec[k]!r} != pool {t[k]!r}')
    if rec.get('hst_cutout_dir') != 'Data/hst_cutouts':
        raise ValueError(f'{t["galaxy_id"]}: Phase 1 used HST cutouts from {rec.get("hst_cutout_dir")}')
    nb = rec.get('nb_veto') or {}
    mode = 'applied' if nb.get('applied') else ('not_applicable' if 'reason' in nb else 'off')
    return {'tag': PHASE1_TAG, 'psf_model': rec['psf_model'], 'psf_fwhm_line': rec['psf_fwhm_line'],
            'psf_moffat_beta': rec.get('psf_moffat_beta'), 'psf_beta_override': rec.get('psf_beta_override'),
            'lsf_model': rec['lsf_model'], 'lsf_sigma_angstrom': rec['lsf_sigma_angstrom'],
            'muse_pixel_scale': rec['muse_pixel_scale'], 'grid_rot_deg': rec.get('grid_rot_deg'),
            'nb_veto': mode, 'nb_merged_labels': nb.get('merged_labels', [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--muselens', type=Path, default=HERE.parent / 'MUSElens')
    ap.add_argument('--musewide', type=Path, default=HERE.parent / 'MUSE-Wide')
    a = ap.parse_args()
    root = a.muselens.resolve()
    pool = json.load(open(root / 'experiments/pool/pool.json'))
    production = {r['galaxy_id'] for r in csv.DictReader(open(root / PRODUCTION_RUN / 'per_galaxy.csv'))}

    (OUT / 'masks').mkdir(parents=True, exist_ok=True)
    targets, manifest = [], {'created': date.today().isoformat(), 'source': str(root), 'phase1_tag': PHASE1_TAG,
                             'targets': {}}
    n_size_rule = 0
    for i, t in enumerate(pool):
        gid = t['galaxy_id']
        sci = root / 'Data/hst_cutouts' / f'{gid}_f814w_sci.fits'
        wht = root / 'Data/hst_cutouts' / f'{gid}_f814w_wht.fits'
        cube = root / 'Data/muse_cutouts' / f'muse_cutout_{gid}.fits'
        for p in (sci, wht, cube):
            if not p.exists():
                raise FileNotFoundError(p)
        h = fits.getheader(sci)
        if h['NAXIS1'] != h['NAXIS2']:
            raise ValueError(f'{gid}: non-square HST cutout')
        rule = 2 * math.ceil(5.0 * t['vdw_r'] * t['vdw_q'] ** 0.5 / 0.03) + 1
        n_size_rule += int(rule == h['NAXIS1'])
        rec_path = root / 'experiments' / f'kinematic_cosi_{PHASE1_TAG}_{gid}' / 'kinematic_cosi_results.json'
        mask_path = rec_path.with_name('hst_mask.npz')
        phase1 = None
        if rec_path.exists():
            phase1 = phase1_block(json.load(open(rec_path)), t)
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)
            shutil.copyfile(mask_path, OUT / 'masks' / f'{gid}.npz')
        if gid in production and phase1 is None:
            raise ValueError(f'{gid}: production galaxy without a Phase 1 record')
        row = dict(t)
        row['request'] = {'hst': {'ra': t['ra'], 'dec': t['dec'], 'size': int(h['NAXIS1']), 'pixel_scale': 0.03},
                          'muse': muse_request(t, cube)}
        row['phase1'] = phase1
        row['in_phase1'] = phase1 is not None
        row['production'] = gid in production
        targets.append(row)
        manifest['targets'][gid] = {
            k: {'hdus': fits_fingerprint(p), 'file_sha256': file_sha256(p), 'bytes': p.stat().st_size}
            for k, p in (('hst_sci', sci), ('hst_wht', wht), ('muse', cube))}
        print(f'[{i + 1}/{len(pool)}] {gid}', flush=True)
    if len(production - {r['galaxy_id'] for r in targets}):
        raise ValueError('production galaxies missing from the pool')

    mw = a.musewide.resolve()
    santini = root / 'experiments/tfr_shape_precision/santini_goodss.tsv'
    skelton = root / 'Data/skelton2014_goodss.tsv'
    catalogs = [
        {'name': 'musewide_dr1_main', 'filename': 'MW_44fields_main_table_v1.0.fits',
         'url': 'https://musewide.aip.de/files/dr1/MW_44fields_main_table_v1.0.fits', 'hash': 'file_sha256',
         'sha256': file_sha256(root / 'Data/MW_44fields_main_table_v1.0.fits'),
         'reference': 'Urrutia et al. 2019, A&A 624, A141'},
        {'name': 'musewide_dr1_emline', 'filename': 'MW_44fields_emline_table_v1.0.fits',
         'url': 'https://musewide.aip.de/files/dr1/MW_44fields_emline_table_v1.0.fits', 'hash': 'file_sha256',
         'sha256': file_sha256(root / 'Data/MW_44fields_emline_table_v1.0.fits'),
         'reference': 'Urrutia et al. 2019, A&A 624, A141'},
        {'name': 'vanderwel2012_table2', 'filename': 'vanderwel2012_table2.dat.gz',
         'url': 'https://cdsarc.cds.unistra.fr/ftp/J/ApJS/203/24/table2.dat.gz', 'hash': 'file_sha256',
         'sha256': file_sha256(root / 'Data/vanderwel_2012/table2.dat.gz'),
         'reference': 'van der Wel et al. 2012, ApJS 203, 24 (GALFIT H-band morphology)'},
        {'name': 'vanderwel2012_readme', 'filename': 'vanderwel2012_ReadMe',
         'url': 'https://cdsarc.cds.unistra.fr/ftp/J/ApJS/203/24/ReadMe', 'hash': 'file_sha256',
         'sha256': file_sha256(root / 'Data/vanderwel_2012/ReadMe'), 'reference': 'VizieR J/ApJS/203/24'},
        {'name': 'santini2015_goodss', 'filename': 'santini2015_goodss.tsv', 'url': vizier_request(santini),
         'hash': 'data_lines_sha256', 'sha256': data_lines_sha256(santini),
         'reference': 'Santini et al. 2015, ApJ 801, 97 (stellar masses)'},
        {'name': 'skelton2014_goodss', 'filename': 'skelton2014_goodss.tsv', 'url': vizier_request(skelton),
         'hash': 'data_lines_sha256', 'sha256': data_lines_sha256(skelton),
         'reference': 'Skelton et al. 2014, ApJS 214, 24 (3D-HST photometry)'},
        {'name': 'hudf_dr2_main', 'filename': 'hudf_dr2_main.csv', 'url': TAP,
         'params': {'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv', 'MAXREC': '1000000',
                    'QUERY': 'SELECT * FROM "J/A+A/670/A4/dr2-main"'},
         'hash': 'file_sha256', 'sha256': file_sha256(mw / 'data/hudf_dr2/dr2_main.csv'),
         'reference': 'Bacon et al. 2023, A&A 670, A4'},
        {'name': 'hudf_dr2_lines', 'filename': 'hudf_dr2_lines.csv', 'url': TAP,
         'params': {'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv', 'MAXREC': '1000000',
                    'QUERY': 'SELECT ID,DataSet,LineSet,Line,IsBlend,DName,Flux,e_Flux,SNR,z,Vdisp,FWHMObs,'
                             'lambdaObs FROM "J/A+A/670/A4/dr2lines"'},
         'hash': 'file_sha256', 'sha256': file_sha256(mw / 'data/hudf_dr2/dr2_lines.csv'),
         'reference': 'Bacon et al. 2023, A&A 670, A4'},
    ]
    json.dump(targets, open(OUT / 'targets.json', 'w'), indent=1)
    json.dump(manifest, open(OUT / 'manifest.json', 'w'), indent=1)
    json.dump(catalogs, open(OUT / 'catalogs.json', 'w'), indent=1)
    print(f'{len(targets)} targets, {sum(r["in_phase1"] for r in targets)} with Phase 1, '
          f'{sum(r["production"] for r in targets)} production; HST size = 5 x r_circ rule for {n_size_rule}')


if __name__ == '__main__':
    main()
