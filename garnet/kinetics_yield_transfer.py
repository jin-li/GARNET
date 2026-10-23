"""Explicit modeling transfer of raw delayed yields between regional estimators.

This helper does not establish physical-domain compatibility. The caller must
separately qualify homologous region ordering, energy groups, delayed families,
and scoring domains. A supplied qualification statement records that modeling
choice; it is not an automatic QA acceptance or covariance inference.
"""
from dataclasses import dataclass
import numpy as np

from .kinetics_data import _array


@dataclass(frozen=True)
class DelayedYieldTransfer:
    """Frozen inputs and transferred coefficients, indexed in caller order.

    Production yields have shape (region, incident group); delayed production
    has shape (region, family, emitted group, incident group), in 1/cm.
    Signed mismatch is receiver minus source. Relative mismatch divides by the
    source yield, with zero for zero/zero channels. No mismatch acceptance
    threshold is defined, and neither field is a statistical uncertainty.
    """
    delayed_production: np.ndarray
    source_total_yields: np.ndarray
    source_delayed_production: np.ndarray
    receiver_total_yields: np.ndarray
    total_yield_difference: np.ndarray
    relative_total_yield_difference: np.ndarray
    source_domain: str
    receiver_domain: str
    qualification: str

    @property
    def max_abs_total_yield_difference(self):
        return float(np.max(np.abs(self.total_yield_difference)))

    @property
    def max_abs_relative_total_yield_difference(self):
        return float(np.max(np.abs(self.relative_total_yield_difference)))


def transfer_delayed_yields(source_total_yields, source_delayed_production,
                            receiver_total_yields, *, source_domain,
                            receiver_domain, qualification):
    """Scale each raw delayed-production column by the receiver total yield.

    Cdel_receiver[n,i,h,g] = Cdel_source[n,i,h,g] / source_total[n,g]
                            * receiver_total[n,g].

    Thus family fractions and emitted spectra within each incident column are
    preserved. This is an opt-in modeling approximation, not direct extraction
    of receiver-domain delayed coefficients. Inputs contain raw production
    yields, never global effective beta. No clipping, fitting or renormalization
    occurs. Positive final coefficients that round to zero are rejected.
    Zero/zero columns are permitted only with zero delayed production;
    a positive receiver yield with zero source yield is always rejected.
    """
    for name,value in (('source_domain',source_domain),('receiver_domain',receiver_domain),
                       ('qualification',qualification)):
        if not isinstance(value,str) or not value.strip():
            raise ValueError(f'{name} must be a nonempty explicit string')
    source = _array(source_total_yields,'source_total_yields')
    delayed = _array(source_delayed_production,'source_delayed_production')
    receiver = _array(receiver_total_yields,'receiver_total_yields')
    if source.ndim != 2 or not all(source.shape) or np.any(source < 0):
        raise ValueError('source_total_yields must be nonnegative nonempty (region,group)')
    N,G = source.shape
    if receiver.shape != (N,G) or np.any(receiver < 0):
        raise ValueError('receiver_total_yields must be nonnegative (region,group) matching source')
    if (delayed.ndim != 4 or delayed.shape[0] != N or delayed.shape[1] == 0
            or delayed.shape[2:] != (G,G) or np.any(delayed < 0)):
        raise ValueError('source_delayed_production must be nonnegative (region,family,target,source)')
    if np.any((source == 0) & (receiver > 0)):
        raise ValueError('zero source denominator with positive receiver total yield')
    with np.errstate(over='ignore',invalid='ignore'):
        column_yield = delayed.sum(axis=(1,2))
    if not np.all(np.isfinite(column_yield)) or np.any(column_yield > source):
        raise ValueError('source delayed column yield exceeds source total yield')
    # Binary exponent scaling avoids an underflowed intermediate fraction or
    # an overflowed intermediate product when the final coefficient is finite.
    # Frexp mantissas keep multiplication/division bounded; ldexp applies the
    # only potentially extreme scaling to the final result.
    delayed_mantissa,delayed_exponent = np.frexp(delayed)
    source_mantissa,source_exponent = np.frexp(source)
    receiver_mantissa,receiver_exponent = np.frexp(receiver)
    yield_mantissa_ratio = np.zeros_like(source)
    np.divide(receiver_mantissa,source_mantissa,out=yield_mantissa_ratio,
              where=source_mantissa > 0)
    mantissa = delayed_mantissa * yield_mantissa_ratio[:,None,None,:]
    exponent = (delayed_exponent + receiver_exponent[:,None,None,:]
                - source_exponent[:,None,None,:])
    with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
        transferred = np.ldexp(mantissa,exponent)
        difference = receiver-source
        relative = np.zeros_like(source)
        np.divide(difference,source,out=relative,where=source > 0)
    positive_result = (delayed > 0) & (receiver[:,None,None,:] > 0)
    if np.any(positive_result & (transferred == 0)):
        raise ValueError('positive transferred delayed production underflows to zero')
    # Extremely separated finite inputs can overflow a descriptive relative
    # mismatch. Reject rather than silently emitting an infinite diagnostic.
    for name,value in (('transferred delayed production',transferred),
                       ('total yield difference',difference),('relative total yield difference',relative)):
        if not np.all(np.isfinite(value)):
            raise ValueError(f'{name} is not representable as a finite real array')
    return DelayedYieldTransfer(
        _array(transferred,'delayed_production'),source,delayed,receiver,
        _array(difference,'total_yield_difference'),_array(relative,'relative_total_yield_difference'),
        source_domain,receiver_domain,qualification)
