"""Bounded absolute branch-XS evaluation for static diffusion/NEM studies.

Scattering is source/emitted; fission matrices are emitted/source. Cross
sections are in cm^-1, D in cm, energy bounds in MeV, temperature in kelvin,
and moderator density in g/cm^3. Interpolation never qualifies physics.
"""
from dataclasses import dataclass, field, replace
from collections.abc import Mapping
from types import MappingProxyType
import json
from pathlib import Path
import numpy as np

from .moderator_paths import TabulatedModeratorPath
from ._microgroup_identity import (MICROGROUP_IDENTITY_KEY, native_identity,
                                   identity_from_bounds, validate_identity)


# Legacy identities retain their original missing/null comparison rules.
# New caller identities belong in COMPATIBILITY_IDENTITIES_NAMESPACE instead.
CLOSURE_IDENTITY_KEYS = (
    'transverse_leakage_identity', 'reflector_mask_sha256',
    'leakage_boundary', 'reference_current_convention',
)
LEGACY_COMPATIBILITY_KEYS = (
    'diffusion', 'absorption_convention', 'scattering_order', 'fission_order',
    'length_unit', 'energy_unit', 'temperature_unit', 'density_unit',
    'adf_prescription', 'geometry_family', 'nuclear_data_family', 'scoring_identity',
    'reference_orders', 'solver_family', 'moderator_path_id', 'library_identity',
    'thermal_scattering_identity', 'energy_structure_identity',
    'library_id', 'nuclear_data_id', 'nuclear_data_library',
    'geometry_id', 'geometry_bounds_sha256', 'scoring_id', 'adf_solver_identity',
    'thermal_scattering_library',
) + CLOSURE_IDENTITY_KEYS
COMPATIBILITY_IDENTITIES_NAMESPACE = 'compatibility_identities'
THERMAL_STAR_REQUIRED_IDENTITIES = (
    'geometry_family', 'nuclear_data_family', 'scoring_identity',
    'adf_prescription', 'solver_family',
)


def _array(value, name, shape=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'biuf':
        raise ValueError(f'{name}: real numeric array required')
    raw = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(raw)) or (shape is not None and raw.shape != shape):
        raise ValueError(f'{name}: finite array with shape {shape} required')
    # Bytes-backed arrays cannot be made writable by their recipient.
    return np.frombuffer(raw.tobytes(), dtype=float).reshape(raw.shape)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biuf':
            raise TypeError('metadata arrays must be numeric')
        if not np.all(np.isfinite(value)):
            raise ValueError('metadata arrays must be finite')
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.generic):
        return _freeze(value.item())
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError('metadata must be finite')
        return value
    raise TypeError(f'unsupported metadata type {type(value).__name__}')


def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _validate_identity_names(value):
    """Reject mapping keys that metadata freezing/JSON would otherwise coerce."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key.strip() for key in value):
            raise ValueError('compatibility_identities requires nonempty string mapping keys')
        for item in value.values():
            _validate_identity_names(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _validate_identity_names(item)


@dataclass(frozen=True)
class NativeBranchState:
    """Unmodified native records, including paired statistical fields/hashes."""
    records: Mapping
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'records', _freeze(self.records))
        object.__setattr__(self, 'metadata', _freeze(self.metadata))


@dataclass(frozen=True)
class BranchState:
    """Converted absolute XS; qualification is explicit caller-supplied evidence.

    ``factors`` use NEM sides x-,y-,x+,y+,z-,z+. ``exterior_mask`` has
    shape (regions,6); marked factors must be exactly unity. No automatic
    Serpent face-order conversion or ADF reconstruction is performed.
    """
    state_id: str
    conditions: Mapping
    region_ids: tuple
    energy_bounds_mev: np.ndarray
    D: np.ndarray
    sigma_a: np.ndarray
    sigma_s: np.ndarray
    fission_matrix: np.ndarray
    chi: np.ndarray
    kappa_fission: np.ndarray | None = None
    total: np.ndarray | None = None
    factors: np.ndarray | None = None
    exterior_mask: np.ndarray | None = None
    metadata: Mapping = field(default_factory=dict)
    qualification: str = 'unvalidated'
    native: NativeBranchState | None = None

    def __post_init__(self):
        if not isinstance(self.state_id, str) or not self.state_id:
            raise ValueError('state_id must be nonempty')
        regions = tuple(self.region_ids)
        if not regions or any(not isinstance(r, str) or not r for r in regions) or len(set(regions)) != len(regions):
            raise ValueError('region_ids must be unique nonempty strings')
        bounds = _array(self.energy_bounds_mev, 'energy_bounds_mev')
        if bounds.ndim != 1 or len(bounds) < 2 or np.any(bounds < 0) or not (np.all(np.diff(bounds) < 0) or np.all(np.diff(bounds) > 0)):
            raise ValueError('energy bounds must be nonnegative and strictly monotone')
        N, G = len(regions), len(bounds)-1
        shapes = {'D': (N,G), 'sigma_a': (N,G), 'sigma_s': (N,G,G),
                  'fission_matrix': (N,G,G), 'chi': (N,G),
                  'kappa_fission': (N,G), 'total': (N,G), 'factors': (N,6,G)}
        object.__setattr__(self, 'region_ids', regions)
        object.__setattr__(self, 'energy_bounds_mev', bounds)
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value is None:
                if name == 'factors':
                    value = np.ones(shape)
                elif name in ('kappa_fission', 'total'):
                    continue
                else:
                    raise ValueError(f'{name} is required')
            arr = _array(value, name, shape)
            if np.any(arr < 0) or (name in ('D', 'factors') and np.any(arr <= 0)):
                raise ValueError(f'{name} must be nonnegative (D/factors positive)')
            object.__setattr__(self, name, arr)
        fuel = self.fission_matrix.sum(axis=(1,2)) > 0
        if np.any(abs(self.chi.sum(axis=1)[fuel]-1) > 1e-12):
            raise ValueError('chi must sum to one in fission-producing regions')
        mask = np.zeros((N,6), dtype=bool) if self.exterior_mask is None else np.asarray(self.exterior_mask)
        if mask.shape != (N,6) or mask.dtype.kind != 'b':
            raise ValueError('exterior_mask must be boolean (regions,6)')
        mask = np.frombuffer(mask.tobytes(), dtype=bool).reshape(mask.shape)
        if np.any(self.factors[mask] != 1):
            raise ValueError('exterior factors must equal unity')
        object.__setattr__(self, 'exterior_mask', mask)
        if not isinstance(self.conditions, Mapping) or any(
                not isinstance(k,str) or not isinstance(v,(str,int,float,bool,np.number))
                or isinstance(v,complex) for k,v in self.conditions.items()):
            raise ValueError('conditions must contain scalar numeric or categorical values')
        for key,value in self.conditions.items():
            if ('temperature' in key or 'density' in key) and (not isinstance(value,(int,float,np.number)) or not np.isfinite(value) or value <= 0):
                raise ValueError(f'{key} must be positive finite')
        object.__setattr__(self, 'conditions', _freeze(self.conditions))
        namespace = COMPATIBILITY_IDENTITIES_NAMESPACE
        if namespace in self.metadata:
            identities = self.metadata[namespace]
            if not isinstance(identities, Mapping):
                raise ValueError('compatibility_identities must be a mapping of nonempty string keys')
            _validate_identity_names(identities)
        if self.native is not None and not isinstance(self.native, NativeBranchState):
            raise TypeError('native must be NativeBranchState')
        declared = self.metadata.get(MICROGROUP_IDENTITY_KEY)
        if MICROGROUP_IDENTITY_KEY in self.metadata:
            declared = validate_identity(declared)
        effective = (native_identity(self.native.records) if self.native is not None else
                     declared if declared is not None else identity_from_bounds(None))
        if declared is not None and declared != effective:
            raise ValueError('microgroup_identity conflicts with retained native records')
        object.__setattr__(self, 'metadata', _freeze({**self.metadata, MICROGROUP_IDENTITY_KEY: effective}))
        if not isinstance(self.qualification, str) or not self.qualification:
            raise ValueError('qualification must be explicit nonempty text')
        expected = {'length_unit': 'cm', 'energy_unit': 'MeV',
                    'temperature_unit': 'K', 'density_unit': 'g/cm3',
                    'scattering_order': 'source-emitted', 'fission_order': 'emitted-source',
                    'absorption_convention': 'reduced', 'diffusion': 'inf'}
        for key, value in expected.items():
            if key in self.metadata and self.metadata[key] != value:
                raise ValueError(f'unsupported {key}: expected {value}')

    @property
    def microgroup_identity(self):
        """Immutable effective native micro-grid identity (explicitly unknown if absent)."""
        return self.metadata[MICROGROUP_IDENTITY_KEY]

    @property
    def nu_sigma_f(self):
        return _array(self.fission_matrix.sum(axis=1), 'nu_sigma_f')

    @property
    def removal(self):
        scatter = self.sigma_s.copy()
        idx = np.arange(scatter.shape[-1])
        scatter[:,idx,idx] = 0
        return _array(self.sigma_a + scatter.sum(axis=2), 'removal')


@dataclass(frozen=True)
class EvaluatedBranchState(BranchState):
    contributing_ids: tuple = ()
    weights: tuple = ()
    interpolated: bool = False

    def __post_init__(self):
        super().__post_init__()
        ids, weights = tuple(self.contributing_ids), tuple(float(w) for w in self.weights)
        if not ids or len(ids) != len(weights) or any(not np.isfinite(w) or w < 0 for w in weights) or abs(sum(weights)-1) > 1e-12:
            raise ValueError('valid contributing_ids and convex weights required')
        object.__setattr__(self, 'contributing_ids', ids)
        object.__setattr__(self, 'weights', weights)


def _arguments(state):
    return {k: getattr(state,k) for k in BranchState.__dataclass_fields__}


class BranchLibrary:
    """Explicit 1D segments or 2D simplices; no extrapolation or implicit grid.

    ``axes`` name numeric conditions. Transforms are ``identity`` (default)
    or ``sqrt`` (the default for fuel_temperature). Every other condition must match exactly, preventing mixing
    configurations, rod/thermal families and other fixed conditions.
    ``compatible_serpent_versions`` optionally declares an application assumption
    for mixing exact version strings; it does not establish equivalence.
    """
    def __init__(self, states, *, axes=('fuel_temperature', 'moderator_density'), transforms=None, simplices=(),
                 compatible_serpent_versions=()):
        if not isinstance(compatible_serpent_versions, (tuple, list)):
            raise ValueError('compatible_serpent_versions must be a tuple or list')
        versions = tuple(compatible_serpent_versions)
        if versions and (len(versions) < 2 or any(not isinstance(v, str) or not v.strip() for v in versions)
                         or len(set(versions)) != len(versions)):
            raise ValueError('compatible_serpent_versions requires at least two distinct nonblank exact strings')
        self.compatible_serpent_versions = versions
        states = tuple(states)
        if not states or any(type(s) is not BranchState for s in states):
            raise ValueError('nonempty original BranchState collection required; evaluated states cannot be table inputs')
        if len({s.state_id for s in states}) != len(states):
            raise ValueError('duplicate state IDs')
        self.states = MappingProxyType({s.state_id:s for s in states})
        self.axes = tuple(axes)
        if len(self.axes) not in (1,2) or len(set(self.axes)) != len(self.axes):
            raise ValueError('one or two distinct axes required')
        transforms = dict({'fuel_temperature': 'sqrt'} if transforms is None and 'fuel_temperature' in self.axes else (transforms or {}))
        if set(transforms)-set(self.axes) or any(v not in ('identity','sqrt') for v in transforms.values()):
            raise ValueError('unsupported transform')
        self.transforms = MappingProxyType({a:transforms.get(a,'identity') for a in self.axes})
        self.simplices = tuple(tuple(ids) for ids in simplices)
        for s in states:
            self._point(s.conditions)
        for ids in self.simplices:
            if len(ids) != len(self.axes)+1 or len(set(ids)) != len(ids):
                raise ValueError('simplex requires distinct dimension+1 states')
            vertices = [self.states[i] for i in ids]
            self._compatible(vertices)
            points = np.array([self._point(s.conditions) for s in vertices])
            if np.linalg.matrix_rank((points[1:]-points[0])) != len(self.axes):
                raise ValueError('degenerate simplex')

    def _point(self, conditions):
        point=[]
        for axis in self.axes:
            value=conditions.get(axis)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(float,int,np.number)) or not np.isfinite(value):
                raise ValueError(f'{axis} must be finite numeric')
            if ('temperature' in axis or 'density' in axis) and value <= 0:
                raise ValueError(f'{axis} must be positive')
            if self.transforms[axis]=='sqrt':
                if value < 0:
                    raise ValueError(f'{axis}: sqrt requires nonnegative value')
                value=np.sqrt(value)
            point.append(float(value))
        return np.array(point)

    def _fixed(self, conditions):
        return {k:v for k,v in conditions.items() if k not in self.axes}

    def _compatible(self, vertices):
        first=vertices[0]
        for s in vertices[1:]:
            if s.microgroup_identity != first.microgroup_identity:
                raise ValueError('incompatible microgroup identities (including known/unknown structures)')
            if self._fixed(s.conditions) != self._fixed(first.conditions):
                raise ValueError('simplex mixes fixed conditions/configurations/families')
            if s.region_ids != first.region_ids or not np.array_equal(s.energy_bounds_mev,first.energy_bounds_mev) or not np.array_equal(s.exterior_mask,first.exterior_mask):
                raise ValueError('incompatible region, energy or exterior topology')
            if any((getattr(s,k) is None)!=(getattr(first,k) is None) for k in ('total','kappa_fission')):
                raise ValueError('inconsistent optional XS channels')
            # Closure presence is significant; legacy non-closure keys retain
            # historical get() equality (omitted and null compare equal).
            if any((key in s.metadata) != (key in first.metadata)
                   for key in CLOSURE_IDENTITY_KEYS):
                raise ValueError('incompatible spatial-reference closure identity declarations')
            if any(_plain(s.metadata.get(key)) != _plain(first.metadata.get(key))
                   for key in LEGACY_COMPATIBILITY_KEYS):
                raise ValueError('incompatible XS units/conventions/ADF prescription')
            namespace = COMPATIBILITY_IDENTITIES_NAMESPACE
            if ((namespace in s.metadata) != (namespace in first.metadata)
                    or _plain(s.metadata.get(namespace)) != _plain(first.metadata.get(namespace))):
                raise ValueError('incompatible caller compatibility identities')
        versions = [s.metadata.get('serpent_version') for s in vertices]
        if any(_plain(v) != _plain(versions[0]) for v in versions[1:]):
            if not self.compatible_serpent_versions or any(
                    not isinstance(v, str) or v not in self.compatible_serpent_versions for v in versions):
                raise ValueError('incompatible Serpent versions; explicit compatibility policy required')

    def _evaluation_metadata(self, vertices):
        metadata = _plain(vertices[0].metadata)
        versions = {s.state_id: _plain(s.metadata.get('serpent_version')) for s in vertices}
        first_version = next(iter(versions.values()))
        if any(v != first_version for v in versions.values()):
            metadata.pop('serpent_version', None)
        metadata.update(source_serpent_versions=versions,
                        compatible_serpent_versions=list(self.compatible_serpent_versions))
        return metadata

    def select(self, state_id):
        s=self.states[state_id]
        data = _arguments(s)
        data['metadata'] = self._evaluation_metadata([s])
        return EvaluatedBranchState(**data, contributing_ids=(s.state_id,), weights=(1.,))

    def evaluate(self, conditions, *, configuration=None, family=None):
        conditions=dict(conditions)
        for key,value in (('configuration',configuration),('family',family)):
            if value is not None:
                if key in conditions and conditions[key] != value:
                    raise ValueError(f'conflicting {key}')
                conditions[key]=value
        point=self._point(conditions)
        fixed=self._fixed(conditions)
        exact=[s for s in self.states.values() if dict(s.conditions)==conditions]
        if len(exact)>1:
            raise ValueError('ambiguous exact conditions; select an explicit state ID')
        if exact:
            return self.select(exact[0].state_id)
        for ids in self.simplices:
            vertices=[self.states[i] for i in ids]
            if self._fixed(vertices[0].conditions)!=fixed:
                continue
            points=np.array([self._point(s.conditions) for s in vertices])
            tail=np.linalg.solve((points[1:]-points[0]).T,point-points[0])
            weights=np.r_[1-tail.sum(),tail]
            # Reject true extrapolation, permit only floating-point boundary noise.
            if np.any(weights < -1e-12) or np.any(weights > 1+1e-12):
                continue
            self._compatible(vertices)
            weights = np.maximum(weights, 0)
            weights /= weights.sum()
            data=_arguments(vertices[0])
            for key in ('D','sigma_a','sigma_s','fission_matrix','chi','kappa_fission','total'):
                if data[key] is not None:
                    data[key]=sum(w*getattr(s,key) for w,s in zip(weights,vertices))
            data['factors']=np.exp(sum(w*np.log(s.factors) for w,s in zip(weights,vertices)))
            data['factors'][vertices[0].exterior_mask]=1
            # Nonfissile chi is harmless metadata; fuel spectra remain normalized.
            chi = data['chi']
            norm = chi.sum(axis=1)
            chi[norm>0]/=norm[norm>0,None]
            data.update(state_id='interpolated:'+','.join(ids),conditions=conditions,native=None,
                        qualification='interpolated-unvalidated',
                        metadata={**self._evaluation_metadata(vertices),'interpolation_domain':list(ids),
                                  'source_qualifications':{s.state_id:s.qualification for s in vertices}})
            return EvaluatedBranchState(**data,contributing_ids=ids,weights=tuple(weights),interpolated=True)
        raise ValueError('request outside explicit interpolation domain or fixed conditions mismatch')

    def evaluate_temperatures(self, fuel_temperature, moderator_temperature, path, *,
                              conditions=None, configuration=None, family=None):
        """Evaluate along an explicit tabulated moderator T/density path.

        Actual moderator temperature is provenance, not an independent axis.
        A supplied density or temperature must agree with that same path.
        """
        if not isinstance(path, TabulatedModeratorPath):
            raise TypeError('path must be TabulatedModeratorPath')
        if self.axes != ('fuel_temperature','moderator_density'):
            raise ValueError('temperature helper requires fuel_temperature/moderator_density axes')
        conditions=dict(conditions or {})
        rho=path.evaluate(moderator_temperature)
        for key,value in (('fuel_temperature',fuel_temperature),
                          ('moderator_temperature',moderator_temperature),('moderator_density',rho)):
            if key in conditions and not np.isclose(conditions[key],value,rtol=0,atol=1e-12):
                raise ValueError(f'{key} conflicts with requested temperature path')
        conditions.pop('moderator_temperature',None)
        conditions.update(fuel_temperature=fuel_temperature,moderator_density=rho)
        result=self.evaluate(conditions,configuration=configuration,family=family)
        expected=result.metadata.get('moderator_path_id')
        if expected is not None and expected != path.path_id:
            raise ValueError('library moderator path identity mismatch')
        return replace(result,metadata={**_plain(result.metadata),
                        'evaluated_moderator_temperature':float(moderator_temperature),
                        'moderator_path_id':path.path_id,'moderator_pressure_bar':path.pressure_bar})

    def save(self, path):
        """Portable JSON (no executable pickle payloads); arrays encoded as lists."""
        payload={'schema':'garnet.branch-xs.v1','axes':self.axes,'transforms':dict(self.transforms),
                 'simplices':self.simplices,'compatible_serpent_versions':self.compatible_serpent_versions,'states':[]}
        for s in self.states.values():
            data=_plain(_arguments(s))
            if s.native is not None:
                data['native']={'records':_plain(s.native.records),'metadata':_plain(s.native.metadata)}
            payload['states'].append(data)
        Path(path).write_text(json.dumps(payload,allow_nan=False,indent=2)+'\n')

    @classmethod
    def load(cls,path):
        payload=json.loads(Path(path).read_text())
        if payload.get('schema') not in ('garnet.branch-xs.v1', 'ndgpu.branch-xs.v1'):
            raise ValueError('unsupported branch library schema')
        states=[]
        for data in payload['states']:
            if data['native'] is not None:
                data['native']=NativeBranchState(**data['native'])
            data['exterior_mask']=np.asarray(data['exterior_mask'],dtype=bool)
            states.append(BranchState(**data))
        return cls(states,axes=payload['axes'],transforms=payload['transforms'],simplices=payload['simplices'],
                   compatible_serpent_versions=payload.get('compatible_serpent_versions', ()))


@dataclass(frozen=True, init=False)
class BenchmarkModeratorPath(TabulatedModeratorPath):
    """Legacy 155-bar PRATIC compatibility wrapper, not an equation of state.

    Historical defaults and the restriction to decreasing density at 155 bar
    are retained. New applications should use ``TabulatedModeratorPath``.
    Changed benchmark knots require a caller-defined path identity.
    """
    temperatures: tuple = (558.15, 573.15, 588.15)
    densities: tuple = (.75572, .72683, .69325)
    path_id: str = 'pratic-155bar-558.15-588.15K-v1'
    pressure_bar: float = 155.

    def __init__(self, temperatures=(558.15, 573.15, 588.15),
                 densities=(.75572, .72683, .69325), pressure_bar=155.,
                 path_id='pratic-155bar-558.15-588.15K-v1'):
        # Preserve the historical positional argument order.
        object.__setattr__(self, 'temperatures', temperatures)
        object.__setattr__(self, 'densities', densities)
        object.__setattr__(self, 'pressure_bar', pressure_bar)
        object.__setattr__(self, 'path_id', path_id)
        self.__post_init__()

    def __post_init__(self):
        super().__post_init__()
        if self.pressure_bar != 155.:
            raise ValueError('benchmark path requires 155 bar')
        if np.any(np.diff(self.densities) >= 0):
            raise ValueError('benchmark path requires decreasing positive density knots')
        if self.path_id == 'pratic-155bar-558.15-588.15K-v1' and (
                self.temperatures != (558.15, 573.15, 588.15)
                or self.densities != (.75572, .72683, .69325)):
            raise ValueError('changed benchmark knots require a new path_id')


def import_serpent_state(state_id, conditions, records, *, diffusion='inf', factors=None,
                         exterior_mask=None, metadata=None, qualification='unvalidated'):
    """Convert explicitly selected region-ID→SerpentXSRecord mapping.

    Raw native fields and source hashes are retained. ADFs must be supplied by
    a separately qualified reconstruction in NEM order; raw DF_SURF_DF is not
    silently substituted. Missing factors mean unity and are recorded.
    """
    if diffusion!='inf':
        raise ValueError("branch importer requires explicit diffusion='inf'")
    conventions={'diffusion':'inf','absorption_convention':'reduced',
                 'scattering_order':'source-emitted','fission_order':'emitted-source',
                 'length_unit':'cm','energy_unit':'MeV','temperature_unit':'K','density_unit':'g/cm3'}
    if any(key in (metadata or {}) and metadata[key] != value for key,value in conventions.items()):
        raise ValueError('import metadata conflicts with native XS units/conventions')
    if not isinstance(records,Mapping) or not records:
        raise ValueError('records must map region IDs to selected Serpent records')
    regions=tuple(records)
    values=list(records.values())
    mats=[r.to_diffusion_material(diffusion='inf') for r in values]
    bounds=values[0].energy_bounds_mev
    if any(not np.array_equal(r.energy_bounds_mev,bounds) for r in values):
        raise ValueError('incompatible native energy structures')
    native=NativeBranchState({rid:{'source':str(r.source),'source_sha256':r.source_sha256,
                                   'record_index':r.record_index,'universe':r.universe,'fields':r.fields}
                              for rid,r in records.items()})
    def stack(name):
        vals=[getattr(m,name) for m in mats]
        if all(v is None for v in vals):
            return None
        if any(v is None for v in vals):
            raise ValueError(f'inconsistent native {name}')
        return np.stack(vals)
    chi = stack('chi')
    nu = stack('nu_sigma_f')
    return BranchState(state_id,conditions,regions,bounds,stack('diffusion'),stack('sigma_a'),
                       stack('sigma_s'),chi[:,:,None]*nu[:,None,:],chi,
                       kappa_fission=stack('kappa_fission'),total=stack('total'),factors=factors,
                       exterior_mask=exterior_mask,native=native,qualification=qualification,
                       metadata={**(metadata or {}),'diffusion':'inf',
                                 'absorption_convention':'reduced','scattering_order':'source-emitted',
                                 'fission_order':'emitted-source','length_unit':'cm','energy_unit':'MeV',
                                 'temperature_unit':'K','density_unit':'g/cm3',
                                 'adf_origin':'unity' if factors is None else 'caller-supplied'})


def build_nem_solver(state, grid, region_indices=None, *, degree=16, **kwargs):
    """Construct a fresh CPU higher-order NEM operator using full fission F.

    ``region_indices`` maps each grid node to a state region index. No operator
    cache is shared between states; grid exterior-factor checks remain active.
    The receiving NEM grid uses three Cartesian axes and six whole-node faces;
    table genericity does not provide arbitrary spatial topology support.
    """
    from .nem_higher_order import NEMHigherOrderDiffusionSolver
    if not isinstance(state,BranchState):
        raise TypeError('state must be BranchState')
    if region_indices is None:
        indices=np.arange(len(state.region_ids))
    else:
        raw=np.asarray(region_indices)
        if raw.dtype.kind not in 'iu':
            raise ValueError('region_indices must be integer indices')
        indices=raw
    if indices.shape!=(grid.n_nodes,) or np.any(indices<0) or np.any(indices>=len(state.region_ids)):
        raise ValueError('region_indices must map every grid node to a valid region')
    return NEMHigherOrderDiffusionSolver(grid,state.D[indices],state.sigma_a[indices],
                 state.sigma_s[indices],state.nu_sigma_f[indices],state.chi[indices],
                 node_face_factors=state.factors[indices],fission_matrix=state.fission_matrix[indices],
                 degree=degree,**kwargs)
