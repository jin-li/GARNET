"""Local homogeneous references using the receiving NEM's exact production.

This reconstructs axis profiles from prescribed means and global face currents.
It does not solve a receiving core or qualify ADF ratios. A constant auxiliary
mean source is reported separately, never added to physical kinetics production.
"""
from dataclasses import dataclass
import numpy as np
from .nem import NEMDiffusionSolver, _PAIRS
from .nem_polynomial_axis import NEMPolynomialAxisResponse
from .kinetics_data import _array
from .face_conventions import NEM_OPPOSITE_FACES
from .nem_degree_policy import (
    DEFAULT_REFERENCE_DEGREES, DEFAULT_REFERENCE_ORDER_TOLERANCE,
    validate_nem_degree, validate_reference_degrees,
)


@dataclass(frozen=True)
class NEMHomogeneousReference:
    face_means: np.ndarray       # N,6,G; uncorrected homogeneous traces
    moments: np.ndarray          # N,3,P-2,G
    coefficients: np.ndarray     # N,3,G,P+1; Legendre in u=2x/L
    coupling: np.ndarray         # R-C/k
    production: np.ndarray       # exact receiving C, target/source
    auxiliary_mean_source: np.ndarray  # M*mean + sum_axis divergence J
    diagnostics: dict


def reconstruct_nem_reference(solver, *, k_eff, means, currents=None,
                              net_inward_integrated=None, degree=None):
    """Reconstruct all local axes with the receiver's degree/leakage policy.

    ``means`` is (N,G), positive. ``currents`` is (F,G), current density in the
    positive global axis on each unique grid face, not outward/inward or an
    integrated current. Alternatively supply signed ``net_inward_integrated``
    (N,6,G) for native region-side tallies; no shared-face averaging occurs.
    Its shared-face disagreement is reported, not repaired or qualified.
    Exactly one current representation is required. Geometry supplies widths
    and face ownership. Transverse
    leakage uses the same quadratic stencil and boundary closure as the solver.
    The supplied k belongs to the reference state; it is not fitted here.

    Local conditioning, weak equations and current/mean/moment recovery are
    checked by the axis response. Negative profiles remain visible in results.
    They are not clipped or admitted as positive ADF denominators. This does
    not assert global vacuum/ADF compatibility of arbitrary prescribed data.
    """
    if not isinstance(solver, NEMDiffusionSolver):
        raise TypeError('a NEM diffusion solver is required')
    solver._assert_unchanged()
    k = _array(k_eff, 'k_eff')
    if k.ndim != 0 or k <= 0:
        raise ValueError('k_eff must be a finite positive scalar')
    mean = _array(means, 'means')
    receiving_degree = getattr(solver,'degree',4)
    p = receiving_degree if degree is None else degree
    p = validate_nem_degree(p, name="reference degree")
    n,g = solver.n_nodes,solver.n_groups
    if mean.shape != (n,g) or np.any(mean <= 0):
        raise ValueError('means must be positive with shape (N,G)')
    grid = solver.grid
    if (currents is None) == (net_inward_integrated is None):
        raise ValueError('supply exactly one current representation')
    if currents is not None:
        current = _array(currents, 'currents')
        if current.shape != (solver.n_faces,g):
            raise ValueError('currents must have shape (F,G) in global-axis density convention')
        side_current = current[grid.node_faces]
        convention = 'shared_face_global_axis_current_density'
    else:
        inward = _array(net_inward_integrated, 'net_inward_integrated')
        if inward.shape != (n,6,g):
            raise ValueError('net_inward_integrated must have shape (N,6,G)')
        side_current = _array(-inward*grid.normal_signs[None,:,None]
                              /grid.node_face_areas[:,:,None], 'side_current')
        convention = 'native_region_side_net_inward_integrated_no_averaging'
    mismatch = 0.
    for low,high in grid.face_nodes:
        if low >= 0 and high >= 0:
            # Shared whole faces have exactly one reciprocal side pair.
            side = np.flatnonzero(grid.neighbors[low] == high)
            for sp in side:
                sm = NEM_OPPOSITE_FACES[sp]
                a,b = side_current[low,sp],side_current[high,sm]
                mismatch = max(mismatch,float(np.max(abs(a-b))))
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        coupling = _array(solver.R-solver.fission/float(k), 'coupling')
        # Net global current difference, before division by transverse width.
        difference = np.stack([side_current[:,sp]-side_current[:,sm]
                               for sm,sp in _PAIRS],axis=1)
        divergence = difference/grid.widths[:,:,None]
        auxiliary = _array(np.einsum('ngh,nh->ng',coupling,mean)+divergence.sum(axis=1),
                           'auxiliary_mean_source')
    faces = np.empty((n,6,g))
    moments = np.empty((n,3,p-2,g))
    coefficients = np.empty((n,3,g,p+1))
    minima = np.empty((n,3,g))
    max_condition = 0.
    errors = {key:0. for key in ('weak_balance_error','current_balance_error',
                                'mean_balance_error','moment_balance_error')}
    for node in range(n):
        for axis,pair in enumerate(_PAIRS):
            samples,weights = grid.leakage_weights(node,axis)
            ell = np.zeros((p-2,g))
            for transverse in range(3):
                if transverse != axis:
                    ell[:2] += weights@difference[samples,transverse]/grid.widths[node,transverse]
            response = NEMPolynomialAxisResponse(solver.D[node],coupling[node],
                        grid.widths[node,axis],degree=p,condition_limit=solver.condition_limit)
            result = response.solve(mean[node],side_current[node,list(pair)],
                                    leakage_moments=ell,rtol=solver.rtol)
            faces[node,list(pair)] = result.face_means
            moments[node,axis] = result.moments
            coefficients[node,axis] = result.coefficients
            minima[node,axis] = result.diagnostics['profile_minima']
            max_condition = max(max_condition,result.diagnostics['condition_number_2_scaled'])
            for key in errors:
                errors[key] = max(errors[key],result.diagnostics[key])
    solver._assert_unchanged()
    diagnostics = dict(errors, polynomial_degree=p, leakage_boundary=grid.leakage_boundary,
                       receiving_polynomial_degree=receiving_degree,
                       solver_signature=solver._configuration_signature.hex(),
                       k_eff=float(k),condition_number_2_scaled_max=max_condition,
                       profile_minima=_array(minima,'profile_minima'),
                       negative_profile_count=int(np.sum(minima<0)),
                       current_convention=convention,
                       shared_face_current_density_max_abs_disagreement=mismatch,
                       auxiliary_source_is_physical=False,ADF_qualification=False)
    return NEMHomogeneousReference(_array(faces,'face_means'),_array(moments,'moments'),
                    _array(coefficients,'coefficients'),coupling,
                    _array(solver.fission,'production'),auxiliary,diagnostics)


@dataclass(frozen=True)
class NEMReferenceFactors:
    factors: np.ndarray
    references: tuple[NEMHomogeneousReference, ...]
    face_change_over_mean: np.ndarray
    factor_relative_change_bound: np.ndarray
    order_tolerance: float
    corrected_trace_relative_error: float


def qualify_nem_reference_factors(solver, *, k_eff, means, heterogeneous_faces,
                                  fuel_mask, degrees=DEFAULT_REFERENCE_DEGREES,
                                  order_tolerance=DEFAULT_REFERENCE_ORDER_TOLERANCE, currents=None,
                                  net_inward_integrated=None):
    """Apply the existing fuel/reflector internal-trace prescription to general C.

    Exactly three increasing reference orders are required; the last must equal
    the receiving order. Fuel profiles at the final order must be positive.
    Both successive face changes must pass the mean-normalized order limit.
    Reflector internal traces use conservative earlier-order positivity and
    factor-change bounds. Negative exterior reflector traces remain visible;
    exterior factors stay unity. This qualifies local factor construction only,
    not a receiving full-core solve or independent transport prediction.
    """
    from .reflector_interface_factors import extend_reflector_interface_factors
    levels = validate_reference_degrees(degrees, getattr(solver, 'degree', 4))
    tolerance = _array(order_tolerance,'order_tolerance')
    if tolerance.ndim!=0 or tolerance<=0:
        raise ValueError('order_tolerance must be finite positive')
    mean = _array(means,'means')
    het = _array(heterogeneous_faces,'heterogeneous_faces')
    fuel = np.asarray(fuel_mask)
    n,g = solver.n_nodes,solver.n_groups
    if het.shape!=(n,6,g) or fuel.dtype.kind!='b' or fuel.shape!=(n,):
        raise ValueError('heterogeneous_faces/fuel_mask must have shape (N,6,G)/(N,)')
    internal = solver.grid.neighbors>=0
    if np.any(het[internal]<=0):
        raise ValueError('internal heterogeneous face means must be positive')
    refs = tuple(reconstruct_nem_reference(solver,k_eff=k_eff,means=mean,
                 currents=currents,net_inward_integrated=net_inward_integrated,
                 degree=degree) for degree in levels)
    final = refs[-1]
    if np.any(final.diagnostics['profile_minima'][fuel]<=0):
        raise ValueError('fuel reference profiles must be positive')
    changes = np.zeros((n,6,g))
    bounds = np.zeros_like(changes)
    for pair in _PAIRS:
        c1 = np.max(abs(refs[1].face_means[:,pair]-refs[0].face_means[:,pair])/mean[:,None,:],axis=(1,2))
        c2 = np.max(abs(refs[2].face_means[:,pair]-refs[1].face_means[:,pair])/mean[:,None,:],axis=(1,2))
        if np.any(np.maximum(c1,c2)>tolerance):
            raise ValueError('reference face-order gate failed')
        for side in pair:
            selected = internal[:,side]&~fuel
            face = final.face_means[selected,side]
            d1,d2 = c1[selected,None]*mean[selected],c2[selected,None]*mean[selected]
            low_middle,low_first = face-d2,face-d2-d1
            if np.any(low_middle<=0) or np.any(low_first<=0):
                raise ValueError('earlier-order internal reflector positivity bound failed')
            changes[selected,side] = np.maximum(c1,c2)[selected,None]
            bounds[selected,side] = np.maximum(face*d1/(low_first*low_middle),d2/low_middle)
    baseline = np.ones_like(het)
    selected = fuel[:,None]&internal
    baseline[selected] = het[selected]/final.face_means[selected]
    if np.any(internal&~fuel[:,None]):
        factors = extend_reflector_interface_factors(baseline,fuel,solver.grid.neighbors,
                    het,final.face_means,changes,bounds,order_tolerance=float(tolerance))
    else:
        factors = _array(baseline,'factors')
    if not np.all(np.isfinite(factors)) or np.any(factors<=0):
        raise ValueError('reference factors must be finite positive')
    corrected = factors[internal]*final.face_means[internal]
    if not np.allclose(corrected,het[internal],rtol=3e-16,atol=0):
        raise ValueError('corrected reference faces differ from heterogeneous traces')
    closure = float(np.max(abs(corrected-het[internal])/het[internal],initial=0.))
    return NEMReferenceFactors(factors,refs,_array(changes,'face_change_over_mean'),
                              _array(bounds,'factor_relative_change_bound'),float(tolerance),closure)
