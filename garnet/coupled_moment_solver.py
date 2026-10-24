"""CPU nonsymmetric power iteration for the tiny coupled moment operator.

Computational validation only. No native-data physical admission, GPU, CG,
regularization, scattering RHS, or alternate solver fallback.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse.linalg import gmres
from .coupled_moments import CoupledMomentOperator
from .moment_materials import _immutable


@dataclass(frozen=True)
class CoupledMomentResult:
    k_eff: float
    moments: np.ndarray
    scalar_flux: np.ndarray
    power: np.ndarray
    converged: bool
    numerical_converged: bool
    reconstructed_nonnegative: bool
    iterations: int
    reason: str
    full_residual: float
    history: tuple
    provenance: dict

    def __post_init__(self):
        for name in ('moments','scalar_flux','power','history','provenance'):
            object.__setattr__(self,name,_immutable(getattr(self,name)))


class CoupledMomentEigenSolver:
    """Fresh source-normalized U iteration using explicit unpreconditioned GMRES.

    max_inner counts GMRES inner iterations (callback_type='legacy'), not restart
    cycles. Each linear residual is recomputed with matrix-free operator action.
    Returned raw scalar/power are never clipped; positivity_tolerance admits only
    roundoff-sized negatives relative to max(1, max absolute field).
    """
    def __init__(self,operator):
        if not isinstance(operator,CoupledMomentOperator):raise ValueError('CoupledMomentOperator required')
        self.operator=operator

    def solve(self,*,initial=None,max_inner=1000,max_outer=200,inner_tolerance=1e-11,
              k_tolerance=1e-10,source_tolerance=1e-9,residual_tolerance=1e-9,
              positivity_tolerance=1e-12):
        op=self.operator
        if any(type(n) is not int or n<1 for n in (max_inner,max_outer)):
            raise ValueError('Positive integer iteration budgets required')
        tolerances=(inner_tolerance,k_tolerance,source_tolerance,residual_tolerance,positivity_tolerance)
        if any(not np.isfinite(t) or t<=0 or t>=1 for t in tolerances):raise ValueError('Finite tolerances in (0,1) required')
        if initial is None:
            u=np.zeros(op.shape);u[:,0,:]=1
        else:
            u=np.array(initial,dtype=float,copy=True)
            op._values(u)
        u=u.reshape(-1)
        def source_size(x):
            phi=op.scalar_flux(x)
            return float(sum(v*np.dot(m.nu_sigma_f,p) for v,m,p in zip(op.mesh.volumes,op.materials,phi)))
        def nonnegative(x):
            return bool(np.all(np.isfinite(x)) and np.min(x)>=-positivity_tolerance*max(1.,float(np.max(np.abs(x)))))
        scalar=op.scalar_flux(u)
        if not nonnegative(scalar):raise ValueError('Initial reconstructed scalar flux is negative')
        source=source_size(u)
        if not np.isfinite(source) or source<=0:raise ValueError('Positive finite initial fission source required')
        u/=source
        if not np.isfinite(np.linalg.norm(op.fission_action(u))) or np.linalg.norm(op.fission_action(u))==0:
            raise ValueError('Nonzero finite fission action required')
        k=1.;history=[];reason='outer_iteration_limit';numerical=False;full=float('inf')
        # Keep the linear operator independent from the mutable assembled sparse
        # matrix so post-GMRES residual checks detect any assembly inconsistency.
        for outer in range(1,max_outer+1):
            rhs=op.fission_action(u);steps=[]
            y,info=gmres(op.assembled,rhs,x0=None,rtol=inner_tolerance,atol=0,
                         restart=min(40,op.size),maxiter=max_inner,callback=steps.append,callback_type='legacy')
            linear=float(np.linalg.norm(op.action(y)-rhs)/np.linalg.norm(rhs))
            record=dict(outer=outer,inner_iterations=len(steps),gmres_info=int(info),linear_residual=linear)
            if info!=0 or not np.isfinite(linear) or linear>inner_tolerance:
                history.append(record);reason='inner_failure_or_full_linear_residual';break
            next_k=source_size(y)
            if not np.isfinite(next_k) or next_k<=0:
                history.append(record);reason='nonpositive_fission_source';break
            candidate=y/next_k
            fnew=op.fission_action(candidate)
            source_error=float(np.linalg.norm(fnew-rhs)/np.linalg.norm(fnew))
            k_error=abs(next_k-k)/abs(next_k)
            residual=op.action(candidate)-fnew/next_k
            denominator=np.linalg.norm(fnew/next_k)
            full=float(np.linalg.norm(residual)/denominator)
            record.update(k_eff=next_k,k_change=k_error,source_change=source_error,full_residual=full)
            history.append(record);u=candidate;k=next_k
            phi=op.scalar_flux(u)
            power=np.stack([m.kappa_fission*p for m,p in zip(op.materials,phi)])
            if not nonnegative(phi) or not nonnegative(power):reason='negative_reconstructed_scalar_or_power';break
            if k_error<=k_tolerance and source_error<=source_tolerance and full<=residual_tolerance:
                numerical=True;reason='converged';break
        phi=op.scalar_flux(u);power=np.stack([m.kappa_fission*p for m,p in zip(op.materials,phi)])
        positive=nonnegative(phi) and nonnegative(power)
        # Recheck the returned pair, including on failed iterations.
        rhs=op.fission_action(u)/k
        full=float(np.linalg.norm(op.action(u)-rhs)/np.linalg.norm(rhs))
        numerical=numerical and np.isfinite(full) and full<=residual_tolerance
        files=(Path(__file__),Path(__file__).with_name('coupled_moments.py'),Path(__file__).with_name('moment_materials.py'),Path(__file__).with_name('spn.py'))
        provenance=dict(physical_admission=False,computational_only=True,solver='CPU scipy GMRES + source-normalized power iteration',
            numpy_version=np.__version__,scipy_version=scipy.__version__,family=op.family,material_provenance=[dict(m.provenance) for m in op.materials],
            implementation_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            max_inner=max_inner,max_outer=max_outer,inner_tolerance=inner_tolerance,k_tolerance=k_tolerance,
            source_tolerance=source_tolerance,residual_tolerance=residual_tolerance,positivity_tolerance=positivity_tolerance,
            source_normalization='sum(cell_volume * NSF dot reconstructed phi0) = 1',
            initial_state_reused=False,regularization=False,fallback=False,scattering_RHS=False)
        return CoupledMomentResult(k,u.reshape(op.shape),phi,power,numerical and positive,numerical,positive,
                                   len(history),reason,full,tuple(history),provenance)
