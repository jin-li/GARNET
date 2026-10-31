"""Full-sphere harmonic projection and native SP scattering; no transport solve."""
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.special import lpmv

from .moment_materials import MomentMaterial

_DEFAULT_BYTES = 256 * 1024**2


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _budget(required, limit):
    limit = _integer(limit, 'max_bytes', 1)
    if required > limit:
        raise ValueError(f'allocation estimate {required} bytes exceeds max_bytes={limit}')


def _readonly(value, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(array)):
        raise ValueError('array must remain finite in storage dtype')
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


def _order(lmax):
    lmax = _integer(lmax, 'lmax')
    if lmax > 7:
        raise ValueError('lmax must be <= 7')
    return lmax


def real_harmonics(directions, lmax, *, max_bytes=_DEFAULT_BYTES):
    """Return H[n,a], ordered (l,m), l=0..L, m=-l..l.

    H00=1 and integral(Ha Hb dOmega)=4*pi*delta_ab. The real basis uses
    Condon--Shortley associated Legendre functions, sine for m<0 and cosine
    for m>0. Input directions must be unit Cartesian vectors.
    """
    lmax = _order(lmax)
    directions = np.asarray(directions)
    if (directions.ndim != 2 or directions.shape[1] != 3 or directions.shape[0] == 0
            or directions.dtype.kind not in 'fiu'):
        raise ValueError('directions must have shape (n_angles, 3) and real values')
    _budget(8 * directions.shape[0] * ((lmax + 1)**2 + 8), max_bytes)
    if not np.all(np.isfinite(directions)) or not np.allclose(
            np.linalg.norm(directions, axis=1), 1, rtol=0, atol=2e-13):
        raise ValueError('directions must be finite unit vectors')
    mu = np.clip(directions[:, 2], -1, 1)
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    basis = np.empty((len(mu), (lmax + 1)**2))
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            k = abs(m)
            norm = math.sqrt((2*l + 1) * math.factorial(l-k) / math.factorial(l+k))
            value = norm * lpmv(k, l, mu)
            if m:
                value = value * math.sqrt(2) * (np.sin(k*phi) if m < 0 else np.cos(k*phi))
            basis[:, l*l + m + l] = value
    return _readonly(basis)


@dataclass(frozen=True)
class ProductQuadrature:
    """Immutable Gauss--Legendre mu × uniform azimuth rule (weights sum 4*pi).

    Even azimuth count supports exact index permutations for x/y/z reflection.
    Counts integrate harmonic products through degree 2L, including L=0.
    """
    lmax: int
    n_polar: int | None = None
    n_azimuth: int | None = None
    max_bytes: int = field(default=_DEFAULT_BYTES, repr=False, compare=False)
    directions: np.ndarray = field(init=False, repr=False, compare=False)
    weights: np.ndarray = field(init=False, repr=False, compare=False)
    harmonics: np.ndarray = field(init=False, repr=False, compare=False)
    orders: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        lmax = _order(self.lmax)
        npolar = lmax + 1 if self.n_polar is None else _integer(self.n_polar, 'n_polar', 1)
        nazi = 2*lmax + 2 if self.n_azimuth is None else _integer(self.n_azimuth, 'n_azimuth', 2)
        if npolar < lmax + 1 or nazi < 2*lmax + 2 or nazi % 2:
            raise ValueError('need n_polar >= L+1 and even n_azimuth >= 2L+2')
        # Includes construction copies and the dense polar eigensystem used by leggauss.
        count = npolar * nazi
        _budget(8 * (4*npolar*npolar + count*(3*(lmax+1)**2 + 24)), self.max_bytes)
        mu, weights = np.polynomial.legendre.leggauss(npolar)
        phi = np.arange(nazi) * (2*np.pi/nazi)
        radial = np.sqrt(1-mu*mu)[:, None]
        directions = np.stack((radial*np.cos(phi), radial*np.sin(phi),
                               np.broadcast_to(mu[:, None], (npolar, nazi))), axis=-1).reshape(-1, 3)
        for name, value in (('lmax', lmax), ('n_polar', npolar), ('n_azimuth', nazi)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'directions', _readonly(directions))
        object.__setattr__(self, 'weights', _readonly(np.repeat(weights, nazi)*(2*np.pi/nazi)))
        object.__setattr__(self, 'harmonics', real_harmonics(directions, lmax, max_bytes=self.max_bytes))
        object.__setattr__(self, 'orders', _readonly(np.repeat(np.arange(lmax+1), 2*np.arange(lmax+1)+1), np.int64))

    @property
    def n_angles(self):
        return self.n_polar * self.n_azimuth

    def reflection(self, axis):
        """Permutation r with directions[r] equal to axis-reflected directions."""
        if axis not in ('x', 'y', 'z'):
            raise ValueError('axis must be x, y, or z')
        polar, azimuth = np.divmod(np.arange(self.n_angles), self.n_azimuth)
        if axis == 'x':
            azimuth = (self.n_azimuth//2 - azimuth) % self.n_azimuth
        elif axis == 'y':
            azimuth = (-azimuth) % self.n_azimuth
        else:
            polar = self.n_polar - 1 - polar
        return _readonly(polar*self.n_azimuth + azimuth, np.int64)


@dataclass(frozen=True)
class NativeAngularSource:
    """Native SP_l source for cell flux psi[cell, angle, from_group].

    Materials must explicitly declare orientation_contract and identical energy
    bounds. Matrices are canonicalized once to [to,from]; no clipping, removal
    inference, fission term, or angular kernel positivity requirement is applied.
    """
    quadrature: ProductQuadrature
    materials: tuple
    cell_material_ids: np.ndarray
    max_bytes: int = field(default=_DEFAULT_BYTES, repr=False, compare=False)
    production: tuple = field(init=False, repr=False, compare=False)
    n_groups: int = field(init=False)
    energy_bounds_mev: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.quadrature, ProductQuadrature):
            raise ValueError('quadrature must be ProductQuadrature')
        materials = tuple(self.materials)
        if not materials or any(not isinstance(m, MomentMaterial) for m in materials):
            raise ValueError('materials must be a nonempty sequence of MomentMaterial')
        ids = np.asarray(self.cell_material_ids)
        if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= len(materials)):
            raise ValueError('cell_material_ids must be a nonempty integer vector indexing materials')
        groups = materials[0].n_groups
        if groups < 1:
            raise ValueError('materials need at least one energy group')
        lcount = self.quadrature.lmax + 1
        _budget(8*(2*len(materials)*lcount*groups*groups + ids.size + groups+1), self.max_bytes)
        canonical = []
        bounds = None
        for material in materials:
            if material.n_groups != groups:
                raise ValueError('energy group count mismatch')
            orientation = material.provenance.get('orientation_contract')
            if orientation not in ('from-major', 'to-major'):
                raise ValueError('explicit orientation_contract required')
            edges = np.asarray(material.provenance.get('energy_bounds_mev'), dtype=float)
            if (edges.shape != (groups+1,) or not np.all(np.isfinite(edges)) or
                    np.any(edges < 0) or not (np.all(np.diff(edges) > 0) or np.all(np.diff(edges) < 0))):
                raise ValueError('finite monotonic energy_bounds_mev required')
            if bounds is not None and not np.array_equal(edges, bounds):
                raise ValueError('energy bounds mismatch')
            bounds = edges
            matrices = []
            for l in range(lcount):
                if l not in material.production:
                    raise ValueError(f'missing production order {l}')
                matrix = np.asarray(material.production[l])
                if matrix.shape != (groups, groups) or matrix.dtype.kind not in 'fiu' or not np.all(np.isfinite(matrix)):
                    raise ValueError(f'production order {l} must be finite real ({groups}, {groups})')
                matrices.append(matrix.T if orientation == 'from-major' else matrix)
            canonical.append(_readonly(np.stack(matrices)))
        object.__setattr__(self, 'materials', materials)
        object.__setattr__(self, 'cell_material_ids', _readonly(ids, np.int64))
        object.__setattr__(self, 'production', tuple(canonical))
        object.__setattr__(self, 'n_groups', groups)
        object.__setattr__(self, 'energy_bounds_mev', _readonly(bounds))

    def _flux(self, angular_flux):
        flux = np.asarray(angular_flux)
        expected = (self.cell_material_ids.size, self.quadrature.n_angles, self.n_groups)
        if flux.shape != expected or flux.dtype.kind not in 'fiu':
            raise ValueError(f'angular_flux must have real shape {expected}')
        cells, angles, groups = expected
        harmonics = (self.quadrature.lmax+1)**2
        # Conservative live workspace: converted input, output, all moments,
        # one cell's transformed moments and weighted projection basis.
        _budget(8*(2*cells*angles*groups + cells*harmonics*groups +
                   2*harmonics*groups + angles*harmonics), self.max_bytes)
        flux = np.asarray(flux, dtype=np.float64)
        if not np.all(np.isfinite(flux)):
            raise ValueError('angular_flux must be finite')
        return flux

    def moments(self, angular_flux):
        """phi[c,a,g] = sum_n w[n] H[n,a] psi[c,n,g]."""
        flux = self._flux(angular_flux)
        q = self.quadrature
        result = np.einsum('na,cng->cag', q.weights[:, None]*q.harmonics, flux, optimize=False)
        if not np.all(np.isfinite(result)):
            raise ValueError('nonfinite angular moments (overflow)')
        return result

    def scatter(self, angular_flux):
        """q[c,n,to] = sum_a H[n,a] (SP_l[a] @ phi[c,a])[to]/(4*pi)."""
        moments = self.moments(angular_flux)
        q = self.quadrature
        result = np.empty((len(self.cell_material_ids), q.n_angles, self.n_groups))
        for cell, material_id in enumerate(self.cell_material_ids):
            transformed = np.empty_like(moments[cell])
            for l in range(q.lmax+1):
                indices = slice(l*l, (l+1)**2)
                transformed[indices] = moments[cell, indices] @ self.production[material_id][l].T
            result[cell] = q.harmonics @ transformed / (4*np.pi)
        if not np.all(np.isfinite(result)):
            raise ValueError('nonfinite scattering source (overflow)')
        return result
