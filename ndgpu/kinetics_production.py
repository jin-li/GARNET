"""Explicit prompt/delayed production split, in emitted/source group order.

This constructs a new total kernel; it does not repair a frozen sampled total
spectrum or claim equivalence to it. No clipping or implicit normalization is
performed. Delayed channels retain their isotope/family identities externally.
"""
from dataclasses import dataclass
import numpy as np
from .kinetics_data import _array


@dataclass(frozen=True)
class KineticsProduction:
    prompt: np.ndarray
    delayed: np.ndarray
    total: np.ndarray


def assemble_kinetics_production(nu_sigma_f, chi_prompt, delayed_production):
    """Build C = chi_prompt*(nuSigmaF - sum_i,target Cdel_i) + sum_i Cdel_i.

    Shapes: nuSigmaF/chi_prompt (N,G); Cdel (N,I,Gtarget,Gsource).
    Units are cm^-1 for production and dimensionless for spectra. Prompt
    spectra must be normalized in every fission-producing node. The explicit
    delayed production must fit within each incident-group total. Exact zeros
    are retained; a negative prompt remainder is rejected, never clipped.
    """
    total_yield = _array(nu_sigma_f, 'nu_sigma_f')
    chi = _array(chi_prompt, 'chi_prompt')
    delayed = _array(delayed_production, 'delayed_production')
    if total_yield.ndim != 2 or 0 in total_yield.shape or chi.shape != total_yield.shape:
        raise ValueError('nu_sigma_f and chi_prompt must have nonempty shape (N,G)')
    n, g = total_yield.shape
    if delayed.ndim != 4 or delayed.shape[0] != n or delayed.shape[1] == 0 or delayed.shape[2:] != (g,g):
        raise ValueError('delayed_production must have shape (N,I,G,G) with I positive')
    if any(np.any(x < 0) for x in (total_yield, chi, delayed)):
        raise ValueError('production and spectra must be nonnegative')
    fuel = np.any(total_yield > 0, axis=1)
    if np.any(abs(chi.sum(axis=1)[fuel] - 1) > 1e-12):
        raise ValueError('chi_prompt must sum to one in fission-producing nodes')
    with np.errstate(over='ignore', invalid='ignore'):
        delayed_total = delayed.sum(axis=1)
        prompt_yield = total_yield - delayed_total.sum(axis=1)
    if not np.all(np.isfinite(prompt_yield)) or np.any(prompt_yield < 0):
        raise ValueError('delayed production exceeds finite total production')
    prompt = chi[:,:,None] * prompt_yield[:,None,:]
    total = prompt + delayed_total
    return KineticsProduction(_array(prompt, 'prompt'), delayed, _array(total, 'total'))
