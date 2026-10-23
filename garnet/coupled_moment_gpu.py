"""Opt-in hybrid GPU linear solves for the full coupled moment operator.

Only assembled CSR matvec and Arnoldi execute on GPU. CPU handles fission,
source normalization, reconstruction and independent matrix-free residuals.
Upstream CuPy 14.2 itself solves the small Hessenberg least squares on CPU.
No CPU solver fallback, fitting, DF, regularization, or scattering RHS.
"""
import hashlib
import inspect
from pathlib import Path
import numpy as np
import scipy
from . import backend
from .coupled_moments import CoupledMomentOperator
from .coupled_moment_solver import CoupledMomentResult

_GMRES_REFERENCE = ('https://raw.githubusercontent.com/cupy/cupy/v14.2.0/'
                    'cupyx/scipy/sparse/linalg/_iterative.py')


def _inner_budget(size, max_inner):
    """Complete restart blocks prevent CuPy's maxiter overshoot."""
    restart = min(40, size, max_inner)
    return restart, (max_inner // restart) * restart


def _check_gmres_api(xp, gmres):
    # Signature alone cannot establish restart/callback accounting semantics.
    # Restrict to the explicitly reviewed upstream 14.2 release series.
    if xp.__version__.split('.')[:2] != ['14', '2']:
        raise RuntimeError('Verified CuPy 14.2 GMRES required; no solver fallback')
    parameters = inspect.signature(gmres).parameters
    if not {'rtol', 'atol', 'restart', 'maxiter', 'callback', 'callback_type'} <= parameters.keys():
        raise RuntimeError('Unsupported CuPy GMRES API; no solver fallback')


class CoupledMomentGPUEigenSolver:
    """GPU CSR transfer once at construction; fresh CPU outer state per solve.

    max_inner is an inner-iteration cap, rounded down to complete restart blocks.
    Inputs and raw immutable results follow CoupledMomentEigenSolver controls.
    GPU/API errors propagate; independent host residuals must meet fixed tolerances.
    """
    def __init__(self, operator, device='gpu'):
        if not isinstance(operator, CoupledMomentOperator):
            raise ValueError('CoupledMomentOperator required')
        if not isinstance(device, str) or device.lower() not in ('gpu', 'cuda', 'rocm', 'hip'):
            raise ValueError("Explicit device='gpu', 'cuda', 'rocm', or 'hip' required")
        device = device.lower()
        xp = backend.get_backend(device)
        import cupy
        if xp is not cupy or xp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('Actual CuPy GPU backend required; no CPU fallback')
        hip = backend.is_rocm(xp)
        if (device == 'cuda' and hip) or (device in ('rocm', 'hip') and not hip):
            raise RuntimeError('Requested GPU device family does not match runtime')
        from cupyx.scipy.sparse import csr_matrix
        from cupyx.scipy.sparse.linalg import gmres
        _check_gmres_api(xp, gmres)
        self.operator = operator
        self.device = device
        self.xp = xp
        self.device_name = backend.device_name(xp)
        self._gmres = gmres
        self._matrix = csr_matrix(operator.assembled)

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
        restart,effective_budget=_inner_budget(op.size,max_inner)
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
            y_gpu,info=self._gmres(self._matrix,self.xp.asarray(rhs),x0=None,
                         rtol=inner_tolerance,atol=0,restart=restart,maxiter=effective_budget,
                         callback=steps.append,callback_type='pr_norm')
            y=self.xp.asnumpy(y_gpu)
            inner_iterations=len(steps)*restart
            if inner_iterations>effective_budget:
                raise RuntimeError('CuPy GMRES exceeded the verified inner budget contract')
            linear=float(np.linalg.norm(op.action(y)-rhs)/np.linalg.norm(rhs))
            record=dict(outer=outer,inner_iterations=inner_iterations,restart_cycles=len(steps),gmres_info=int(info),linear_residual=linear)
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
        files=(Path(__file__),Path(__file__).with_name('coupled_moment_solver.py'),Path(__file__).with_name('backend.py'),Path(__file__).with_name('coupled_moments.py'),Path(__file__).with_name('moment_materials.py'),Path(__file__).with_name('spn.py'))
        provenance=dict(physical_admission=False,computational_only=True,solver='GPU CuPy GMRES + CPU source-normalized power iteration',
            cupy_version=self.xp.__version__,device_requested=self.device,device_actual=self.device_name,
            gpu_work='CSR sparse matvec and Arnoldi',
            cpu_work='fission/source updates, normalization, physical reconstruction, independent residuals; CuPy Hessenberg least squares',
            assembled_matrix_transfers=1,gmres_restart=restart,effective_inner_budget=effective_budget,
            unused_inner_budget=max_inner-effective_budget,
            inner_iteration_accounting='CuPy 14.2 pr_norm restart callbacks times restart',
            gmres_reference=_GMRES_REFERENCE,
            numpy_version=np.__version__,scipy_version=scipy.__version__,family=op.family,material_provenance=[dict(m.provenance) for m in op.materials],
            implementation_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            max_inner=max_inner,max_outer=max_outer,inner_tolerance=inner_tolerance,k_tolerance=k_tolerance,
            source_tolerance=source_tolerance,residual_tolerance=residual_tolerance,positivity_tolerance=positivity_tolerance,
            source_normalization='sum(cell_volume * NSF dot reconstructed phi0) = 1',
            initial_state_reused=False,regularization=False,fallback=False,scattering_RHS=False)
        return CoupledMomentResult(k,u.reshape(op.shape),phi,power,numerical and positive,numerical,positive,
                                   len(history),reason,full,tuple(history),provenance)
