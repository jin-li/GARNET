"""Report precursor families without replacing distinct physical decay channels.

For ENDF data, isotope A/family 1 and isotope B/family 1 need not have the
same decay constant. They can be retained as separate production channels in
the static NEM projection and grouped only for reporting afterwards.
"""
from dataclasses import dataclass
import numpy as np


def _weighted_decay(beta, decay):
    total = float(np.sum(beta))
    if total == 0:
        return None
    active = beta > 0
    beta, decay = beta[active], decay[active]
    scale = float(np.max(decay))
    value = float(scale * np.sum((beta/total)*(decay/scale)))
    if not np.isfinite(value) or value <= 0:
        raise ValueError('nonfinite or nonpositive weighted decay')
    return value


@dataclass(frozen=True)
class DelayedFamilySummary:
    """Importance-weighted reporting reduction, preserving every input channel.

    ``family_decay`` is a production/importance-weighted arithmetic summary,
    not an exact replacement for several different precursor poles. An empty
    production family has decay ``None``. All fractions are dimensionless,
    and all decay rates are in s^-1.
    """
    channel_ids: tuple[str, ...]
    channel_beta: tuple[float, ...]
    channel_decay: tuple[float, ...]
    channel_reported_family: tuple[str, ...]
    family_ids: tuple[str, ...]
    family_beta: tuple[float, ...]
    family_decay: tuple[float | None, ...]
    family_single_pole_exact: tuple[bool, ...]

    @property
    def beta_total(self):
        return float(sum(self.channel_beta))

    @property
    def lambda_score_weighted(self):
        """Arithmetic mean with the original channel beta weights."""
        return _weighted_decay(np.asarray(self.channel_beta), np.asarray(self.channel_decay))

    @property
    def lambda_low_frequency(self):
        """Reduction preserving sum(beta_channel/lambda_channel), not grouped poles."""
        beta, decay = np.asarray(self.channel_beta), np.asarray(self.channel_decay)
        if self.beta_total == 0:
            return None
        active = beta > 0
        beta, decay = beta[active], decay[active]
        minimum = float(np.min(decay))
        denominator = float(np.sum((beta/self.beta_total)*(minimum/decay)))
        value = minimum/denominator
        if not np.isfinite(value) or value <= 0:
            raise ValueError('nonfinite or nonpositive low-frequency decay')
        return value


def summarize_delayed_channels(beta, decay, *, channel_ids, reported_families):
    """Group projected channels using an explicit native-family mapping.

    Supply the beta and decay vectors returned by a physical projection whose
    input channels retain isotope (or region) identity. ``reported_families``
    assigns each channel to its native family; matching labels are a declared
    mapping, not inferred by sorting decay constants. Original channel values
    remain available. No regional beta is constructed from effective values.

    The grouped values obey beta_i=sum_c beta_c and
    lambda_i=sum_c beta_c*lambda_c/sum_c beta_c. This is a static reporting
    reduction. A different-decay mixture is not an exact transient family.
    """
    def vector(value, name):
        raw = np.asarray(value)
        if raw.dtype.kind not in 'iuf' or raw.ndim != 1 or not raw.size:
            raise ValueError(f'{name} must be a nonempty real vector')
        result = np.asarray(raw,dtype=float)
        if not np.all(np.isfinite(result)):
            raise ValueError(f'{name} must be finite')
        return result
    beta, decay = vector(beta,'beta'), vector(decay,'decay')
    if beta.shape != decay.shape or np.any(beta < 0) or beta.sum() >= 1 or np.any(decay <= 0):
        raise ValueError('compatible nonnegative beta (sum < 1) and positive decay required')
    def labels(value, name):
        if isinstance(value,str):
            raise ValueError(f'{name} must contain one string per channel')
        result = tuple(value)
        if len(result) != len(beta) or any(not isinstance(x,str) or not x for x in result):
            raise ValueError(f'{name} must contain one nonempty string per channel')
        return result
    ids = labels(channel_ids,'channel_ids')
    family = labels(reported_families,'reported_families')
    if len(set(ids)) != len(ids):
        raise ValueError('channel_ids must be unique')
    names = tuple(dict.fromkeys(family))
    fb, fd, exact = [], [], []
    for name in names:
        mask = np.array([x == name for x in family])
        active = decay[mask & (beta > 0)]
        fb.append(float(beta[mask].sum()))
        fd.append(_weighted_decay(beta[mask],decay[mask]))
        exact.append(bool(len(active) <= 1 or np.all(active == active[0])))
    return DelayedFamilySummary(ids,tuple(map(float,beta)),tuple(map(float,decay)),family,
                                names,tuple(fb),tuple(fd),tuple(exact))
