"""Standalone access to the data used by MUSElens: target catalogue, HST F814W and MUSE cutouts, PSF/LSF, WCS,
masks and weight maps, fetched from the public archives and checked against the files MUSElens used."""
from muselens_data.fetch import fetch_catalogs, fetch_hst, fetch_muse, fetch_target
from muselens_data.load import (hst_variance, load_hst, load_muse, prepare_line_window, regenerate_hst_mask)
from muselens_data.masks import load_frozen_hst_mask
from muselens_data.targets import get_target, load_targets, target_ids
from muselens_data.verify import verify_catalogs, verify_target

__version__ = '1.0.0'

__all__ = ['fetch_catalogs', 'fetch_hst', 'fetch_muse', 'fetch_target', 'hst_variance', 'load_hst', 'load_muse',
           'prepare_line_window', 'regenerate_hst_mask', 'load_frozen_hst_mask', 'get_target', 'load_targets',
           'target_ids', 'verify_catalogs', 'verify_target']
