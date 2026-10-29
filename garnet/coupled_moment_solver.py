"""CPU nonsymmetric power iteration for the tiny coupled moment operator.

Computational validation only. No native-data physical admission, GPU, CG,
regularization, scattering RHS, or alternate solver fallback.
"""
from dataclasses import dataclass
import hashlib
from numbers import Real
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator, gmres, spilu
from .coupled_moments import CoupledMomentOperator
from .corrected_native_moments import CorrectedNativeMomentOperator
from .moment_materials import _immutable


def _group_principal_block(matrix, groups, group, *, max_rows=4096, max_nnz=1_000_000):
    """Copy one native energy group's principal block with bounded workspace.

    Native flat indices for a group are ``G*k+g`` for all cell/moment pairs k.
    Slice at most max_rows selected rows and max_nnz source entries at once
    (one indivisible larger row is allowed). Count first, then fill the exact
    final CSR allocation; never retain selected rows across all source columns.
    Explicit zeros, duplicates and within-row entry order are preserved.
    """
    for value, label in ((groups,'groups'), (max_rows,'max_rows'), (max_nnz,'max_nnz')):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<1:
            raise ValueError(label+' must be a positive integer')
    if isinstance(group,(bool,np.bool_)) or not isinstance(group,(int,np.integer)) or not 0<=group<groups:
        raise ValueError('group must be an integer in [0, groups)')
    if not scipy.sparse.isspmatrix_csr(matrix) or matrix.shape[0]!=matrix.shape[1] or matrix.shape[0]%groups:
        raise ValueError('Square native CSR matrix with size divisible by groups required')
    rows=np.arange(group,matrix.shape[0],groups,dtype=np.int64)
    dimension=len(rows)
    def chunks():
        start=0
        while start<dimension:
            stop=min(start+max_rows,dimension)
            selected=rows[start:stop]
            counts=matrix.indptr[selected+1]-matrix.indptr[selected]
            cumulative=np.cumsum(counts,dtype=np.int64)
            count=max(1,int(np.searchsorted(cumulative,max_nnz,side='right')))
            stop=start+count
            yield start,stop,matrix[rows[start:stop],:]
            start=stop
    # int64 counts prevent overflow before choosing the final CSR index width.
    indptr=np.empty(dimension+1,dtype=np.int64);indptr[0]=0
    for start,stop,chunk in chunks():
        keep=chunk.indices%groups==group
        cumulative=np.empty(chunk.nnz+1,dtype=np.int64);cumulative[0]=0
        np.cumsum(keep,dtype=np.int64,out=cumulative[1:])
        indptr[start+1:stop+1]=indptr[start]+cumulative[chunk.indptr[1:]]
        del chunk,keep,cumulative
    nnz=int(indptr[-1])
    index_dtype=np.int32 if max(dimension,nnz)<=np.iinfo(np.int32).max else np.int64
    indices=np.empty(nnz,dtype=index_dtype)
    data=np.empty(nnz,dtype=matrix.dtype)
    for start,stop,chunk in chunks():
        keep=chunk.indices%groups==group
        lo,hi=int(indptr[start]),int(indptr[stop])
        indices[lo:hi]=chunk.indices[keep]//groups
        data[lo:hi]=chunk.data[keep]
        del chunk,keep
    return scipy.sparse.csr_matrix((data,indices,indptr.astype(index_dtype,copy=False)),
                                   shape=(dimension,dimension),copy=False)


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
    """Fresh source-normalized U iteration with CPU GMRES and optional fixed ILU.

    max_inner counts GMRES inner iterations (callback_type='legacy'), not restart
    cycles. Each linear residual is recomputed with matrix-free operator action.
    Returned raw scalar/power are never clipped; positivity_tolerance admits only
    roundoff-sized negatives relative to max(1, max absolute field).
    preconditioner='ilu' freezes a CSC copy and factors it once per solve with
    drop_tol=1e-6 and fill_factor=20. Setup failure raises; no fallback occurs.
    preconditioner='group_ilu' instead freezes one all-moment, all-cell block
    per energy group and applies their factors as fixed block Jacobi. GMRES
    retains the full assembled matrix, including every off-group coefficient.
    group_ilu_fill_factor explicitly budgets only group ILU (default 20);
    group_ilu_setup_callback receives copied setup statistics, never factors.
    iteration_callback receives scalar progress snapshots. SciPy's monitored
    preconditioned relative residual is not the true action acceptance residual.
    """
    def __init__(self,operator):
        if not isinstance(operator,(CoupledMomentOperator,CorrectedNativeMomentOperator)):
            raise ValueError('Explicit native coupled or corrected-native operator required')
        self.operator=operator

    def solve(self,*,initial=None,max_inner=1000,max_outer=200,inner_tolerance=1e-11,
              k_tolerance=1e-10,source_tolerance=1e-9,residual_tolerance=1e-9,
              positivity_tolerance=1e-12,preconditioner='none',group_ilu_fill_factor=20,
              group_ilu_setup_callback=None,iteration_callback=None):
        op=self.operator
        if preconditioner not in ('none','ilu','group_ilu'):
            raise ValueError("preconditioner must be 'none', 'ilu' or 'group_ilu'")
        if isinstance(group_ilu_fill_factor,(bool,np.bool_)) or not isinstance(group_ilu_fill_factor,Real):
            raise ValueError('group_ilu_fill_factor must be finite numeric and at least 1')
        try:
            group_fill=float(group_ilu_fill_factor)
        except OverflowError as exc:
            raise ValueError('group_ilu_fill_factor must be finite numeric and at least 1') from exc
        if not np.isfinite(group_fill) or group_fill<1:
            raise ValueError('group_ilu_fill_factor must be finite numeric and at least 1')
        if group_ilu_setup_callback is not None and not callable(group_ilu_setup_callback):
            raise ValueError('group_ilu_setup_callback must be callable or None')
        if iteration_callback is not None and not callable(iteration_callback):
            raise ValueError('iteration_callback must be callable or None')
        # Corrected MeshGrid volumes are exposed explicitly by the operator.
        volumes=op.volumes if isinstance(op,CorrectedNativeMomentOperator) else op.mesh.volumes
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
            return float(sum(v*np.dot(m.nu_sigma_f,p) for v,m,p in zip(volumes,op.materials,phi)))
        def nonnegative(x):
            return bool(np.all(np.isfinite(x)) and np.min(x)>=-positivity_tolerance*max(1.,float(np.max(np.abs(x)))))
        scalar=op.scalar_flux(u)
        if not nonnegative(scalar):raise ValueError('Initial reconstructed scalar flux is negative')
        source=source_size(u)
        if not np.isfinite(source) or source<=0:raise ValueError('Positive finite initial fission source required')
        u/=source
        if not np.isfinite(np.linalg.norm(op.fission_action(u))) or np.linalg.norm(op.fission_action(u))==0:
            raise ValueError('Nonzero finite fission action required')
        solve_matrix=op.assembled;M=None
        preconditioner_metadata=dict(setup_seconds=0.,matrix_nnz=int(op.assembled.nnz))
        if preconditioner=='ilu':
            started=perf_counter()
            solve_matrix=op.assembled.tocsc(copy=True)
            try:
                factor=spilu(solve_matrix,drop_tol=1e-6,fill_factor=20)
            except (RuntimeError,ValueError) as exc:
                raise RuntimeError('ILU factorization failed; no solver fallback') from exc
            M=LinearOperator(solve_matrix.shape,matvec=factor.solve,dtype=solve_matrix.dtype)
            preconditioner_metadata.update(setup_seconds=perf_counter()-started,
                matrix_nnz=int(solve_matrix.nnz),drop_tol=1e-6,fill_factor=20,
                superlu_nnz=int(factor.nnz),
                superlu_nnz_caveat='SuperLU.nnz reports internal factor storage, not an explicit L/U CSC entry count')
        elif preconditioner=='group_ilu':
            started=perf_counter()
            cells,moments,groups=op.shape
            group_factors=[];group_stats=[]
            def setup_event(phase,group,**details):
                if group_ilu_setup_callback is not None:
                    group_ilu_setup_callback(dict(phase=phase,group=group,fill_factor=group_fill,
                        completed_groups=[dict(stat) for stat in group_stats],**details))
            def setup_failure(group,phase,exc,current=None):
                details=dict(failed_phase=phase,error_type=type(exc).__name__,error=str(exc))
                if current is not None:details['current_group_stats']=dict(current)
                if isinstance(exc,MemoryError):
                    failure=MemoryError(f'Group ILU {"block extraction" if phase=="extraction" else phase} failed for group {group} after {len(group_factors)} completed factors; no solver fallback')
                else:
                    failure=RuntimeError(f'Group ILU {"block extraction" if phase=="extraction" else phase} failed for group {group}; no solver fallback')
                failure.args=(str(failure)+f'; fill_factor={group_fill}; completed_groups={group_stats}',)
                failure.group=group;failure.fill_factor=group_fill
                failure.completed_groups=[dict(stat) for stat in group_stats]
                try:
                    setup_event('failure',group,**details)
                except Exception as callback_exc:
                    # A failed logger must not replace the numerical cause.
                    failure.callback_error=str(callback_exc)
                return failure
            for group in range(groups):
                group_started=perf_counter()
                # Native layout is (cells, M, G): each group block retains
                # every moment of each cell, including SP5's third moment.
                setup_event('before_extraction',group)
                try:
                    indices=(moments*groups*np.arange(cells)[:,None]+groups*np.arange(moments)[None,:]+group).ravel()
                    block=_group_principal_block(solve_matrix,groups,group).tocsc(copy=True)
                except (MemoryError,RuntimeError,ValueError) as exc:
                    raise setup_failure(group,'extraction',exc) from exc
                current=dict(group=group,dimension=int(block.shape[0]),matrix_nnz=int(block.nnz))
                setup_event('before_factorization',group,current_group_stats=dict(current))
                try:
                    factor=spilu(block,drop_tol=1e-6,fill_factor=group_fill)
                except (MemoryError,RuntimeError,ValueError) as exc:
                    raise setup_failure(group,'factorization',exc,current) from exc
                group_factors.append((indices,factor))
                current.update(superlu_nnz=int(factor.nnz),setup_seconds=perf_counter()-group_started)
                group_stats.append(current)
                del block
                setup_event('after_factorization',group,current_group_stats=dict(current))
            # Close over the factors once: neither later matrix mutation nor
            # outer iterations rebuild or update the preconditioner.
            group_factors=tuple(group_factors)
            def group_solve(rhs):
                result=np.empty(op.size,dtype=solve_matrix.dtype)
                for indices,factor in group_factors:
                    result[indices]=factor.solve(np.asarray(rhs).reshape(-1)[indices])
                return result
            M=LinearOperator(solve_matrix.shape,matvec=group_solve,dtype=solve_matrix.dtype)
            preconditioner_metadata.update(setup_seconds=perf_counter()-started,
                drop_tol=1e-6,fill_factor=group_fill,group_ilu_fill_factor=group_fill,groups=group_stats,
                group_block_extraction='two-pass bounded CSR rows; at most 4096 rows and 1000000 source entries per chunk, except one larger row',
                group_cell_ordering=('ascending cells; moments 0,1 per cell; flat indices [2*G*i+g, 2*G*i+G+g]' if moments==2 else
                                     f'ascending cells; moments 0..{moments-1} per cell; flat indices M*G*i+G*m+g for every m'),
                superlu_nnz_caveat='SuperLU.nnz reports internal factor storage, not an explicit L/U CSC entry count')
        k=1.;history=[];reason='outer_iteration_limit';numerical=False;full=float('inf')
        # Keep the linear operator independent from the mutable assembled sparse
        # matrix so post-GMRES residual checks detect any assembly inconsistency.
        for outer in range(1,max_outer+1):
            rhs=op.fission_action(u);steps=[]
            callback=steps.append
            if iteration_callback is not None:
                iteration_callback(dict(phase='inner_start',outer=outer,max_inner=max_inner,
                    restart=min(40,op.size),inner_tolerance=float(inner_tolerance)))
                def callback(value):
                    steps.append(value)
                    iteration_callback(dict(phase='inner_iteration',outer=outer,inner_iteration=len(steps),
                        scipy_preconditioned_residual_norm=float(value),
                        residual_kind='scipy_preconditioned_relative_residual_monitor_only',used_for_acceptance=False))
            y,info=gmres(solve_matrix,rhs,x0=None,rtol=inner_tolerance,atol=0,M=M,
                         restart=min(40,op.size),maxiter=max_inner,callback=callback,callback_type='legacy')
            linear=float(np.linalg.norm(op.action(y)-rhs)/np.linalg.norm(rhs))
            record=dict(outer=outer,inner_iterations=len(steps),gmres_info=int(info),linear_residual=linear)
            if iteration_callback is not None:
                iteration_callback(dict(phase='inner_complete',outer=outer,count=len(steps),gmres_info=int(info),
                    true_action_relative_residual=linear,accepted_linear=bool(info==0 and np.isfinite(linear) and linear<=inner_tolerance)))
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
            def outer_event(positive,converged,event_reason):
                if iteration_callback is not None:
                    iteration_callback(dict(phase='outer_complete',outer=outer,k_eff=float(next_k),k_change=float(k_error),
                        source_change=source_error,full_residual=full,reconstructed_nonnegative=positive,
                        converged=converged,reason=event_reason))
            if not nonnegative(phi) or not nonnegative(power):
                reason='negative_reconstructed_scalar_or_power';outer_event(False,False,reason);break
            if k_error<=k_tolerance and source_error<=source_tolerance and full<=residual_tolerance:
                numerical=True;reason='converged';outer_event(True,True,reason);break
            outer_event(True,False,'outer_iteration_limit' if outer==max_outer else 'continue')
        phi=op.scalar_flux(u);power=np.stack([m.kappa_fission*p for m,p in zip(op.materials,phi)])
        positive=nonnegative(phi) and nonnegative(power)
        # Recheck the returned pair, including on failed iterations.
        rhs=op.fission_action(u)/k
        full=float(np.linalg.norm(op.action(u)-rhs)/np.linalg.norm(rhs))
        numerical=numerical and np.isfinite(full) and full<=residual_tolerance
        files=(Path(__file__),Path(__file__).with_name('coupled_moments.py'),Path(__file__).with_name('moment_materials.py'),Path(__file__).with_name('spn.py'))
        if isinstance(op,CorrectedNativeMomentOperator):
            files += tuple(Path(__file__).with_name(name) for name in
                           ('corrected_native_moments.py','corrected_mesh.py','mesh.py'))
        provenance=dict(physical_admission=False,computational_only=True,solver='CPU scipy GMRES + source-normalized power iteration',
            numpy_version=np.__version__,scipy_version=scipy.__version__,family=op.family,moments_per_cell=op.n_moments,
            required_orders=op.required_orders,material_provenance=[dict(m.provenance) for m in op.materials],
            implementation_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            max_inner=max_inner,max_outer=max_outer,inner_tolerance=inner_tolerance,k_tolerance=k_tolerance,
            source_tolerance=source_tolerance,residual_tolerance=residual_tolerance,positivity_tolerance=positivity_tolerance,
            group_ilu_fill_factor=group_fill if preconditioner=='group_ilu' else None,
            source_normalization='sum(cell_volume * NSF dot reconstructed phi0) = 1',
            initial_state_reused=False,regularization=False,fallback=False,scattering_RHS=False,
            preconditioner=preconditioner,preconditioner_metadata=preconditioner_metadata)
        return CoupledMomentResult(k,u.reshape(op.shape),phi,power,numerical and positive,numerical,positive,
                                   len(history),reason,full,tuple(history),provenance)
