"""Native moment data and full coupled loss matrices; no transport solve.

Higher-order Serpent estimators are not thereby consistent with deterministic
transport theory. Orientation is a caller contract, never inferred from data.
"""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np


def _immutable(value):
    if isinstance(value, dict):
        return MappingProxyType({k:_immutable(v) for k,v in value.items()})
    if isinstance(value, (list,tuple)):
        return tuple(_immutable(v) for v in value)
    if isinstance(value, np.ndarray):
        # Bytes-backed arrays cannot be made writable again via setflags.
        return np.frombuffer(value.astype(np.float64).tobytes(), dtype=np.float64).reshape(value.shape)
    return value


@dataclass(frozen=True)
class MomentMaterial:
    """Immutable native matrices in the explicitly declared input orientation.

    loss[l][to,from] = diag(total) - SP_l[to,from]. Collision S_l is
    retained independently. All printed pairs remain native; no SE is inferred.
    Full Sigma0 includes diagonal scattering cancellation exactly once.
    """
    total: np.ndarray
    chi: np.ndarray
    nu_sigma_f: np.ndarray
    kappa_fission: np.ndarray
    collision: dict
    production: dict
    loss: dict
    raw: dict
    reported_errors: dict
    provenance: dict
    balance_diagnostics: dict

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self,name,_immutable(getattr(self,name)))

    @property
    def n_groups(self):
        return self.total.size


def moment_material_from_record(record, *, orientation, orders=None):
    """Construct only after an explicit from-major/to-major flattening contract.

    No inversion, diagonal approximation, correction, fitting or regularization.
    Chi remains the native estimate, including all-zero nonfissile spectra.
    """
    from .serpent_xs import SerpentMomentRecord, _moment_orders
    if not isinstance(record,SerpentMomentRecord):
        raise ValueError('Explicit opt-in moment record required')
    if orientation not in ('from-major','to-major'):
        raise ValueError('Explicit orientation must be from-major or to-major')
    record._validate()
    orders = _moment_orders(record.required_orders if orders is None else orders)
    g=record.n_groups
    raw={}; errors={}; collision={}; production={}; loss={}
    for field in ('INF_TOT','INF_NSF','INF_RABSXS','INF_FISS','INF_KAPPA'):
        estimate=record.estimate(field)
        raw[field]=estimate.mean;errors[field]=estimate.relative_error
    if 'INF_CHIT' in record.fields:
        estimate=record.estimate('INF_CHIT');raw['INF_CHIT']=estimate.mean;errors['INF_CHIT']=estimate.relative_error
    else:
        raw['INF_CHIT']=np.zeros(g)
    for order in orders:
        for kind, target in (('S',collision),('SP',production)):
            estimate=record.moment_estimate(kind,order)
            key=f'INF_{kind}{order}'
            raw[key]=estimate.mean;errors[key]=estimate.reported_error
            target[order]=estimate.mean
        operator=production[order].T if orientation=='from-major' else production[order]
        loss[order]=np.diag(raw['INF_TOT'])-operator
    sp0=production[0] if orientation=='from-major' else production[0].T
    residual=raw['INF_TOT']-raw['INF_RABSXS']-sp0.sum(axis=1)
    return MomentMaterial(raw['INF_TOT'],raw['INF_CHIT'],raw['INF_NSF'],raw['INF_FISS']*raw['INF_KAPPA'],
        collision,production,loss,raw,errors,
        dict(source=str(record.source),source_sha256=record.source_sha256,record_index=record.record_index,
             universe=record.universe,version=record.version,energy_bounds_mev=record.energy_bounds_mev,
             orders=orders,orientation_contract=orientation,physical_admission=False,
             higher_estimator_transport_consistency_validated=False,reported_error_sign_interpreted=False,
             covariance_inferred=False,local_availability_not_general_version_support=True,
             chi_present='INF_CHIT' in record.fields),
        dict(total_minus_reduced_absorption_production_out=residual))
