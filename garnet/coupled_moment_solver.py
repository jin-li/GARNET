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
    preconditioner='group_ilu_coarse' adds a fixed connected-aggregate native
    correction to those same group factors; no default or acceptance changes.
    inner_warm_start=True keeps the first linear solve cold, then starts at
    k*u using the previous source-normalized candidate and current eigenvalue.
    accepted_outer_state_callback receives copied linear/native states only
    after linear acceptance and candidate positivity; it may stop by raising.
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
              group_ilu_setup_callback=None,iteration_callback=None,inner_warm_start=False,
              accepted_outer_state_callback=None):
        op=self.operator
        if type(inner_warm_start) is not bool:
            raise ValueError('inner_warm_start must be a bool')
        if preconditioner not in ('none','ilu','group_ilu','group_ilu_coarse'):
            raise ValueError("preconditioner must be 'none', 'ilu', 'group_ilu' or 'group_ilu_coarse'")
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
        if accepted_outer_state_callback is not None and not callable(accepted_outer_state_callback):
            raise ValueError('accepted_outer_state_callback must be callable or None')
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
        elif preconditioner in ('group_ilu','group_ilu_coarse'):
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
            if preconditioner=='group_ilu_coarse':
                from .native_coarse import NativeCoarsePreconditioner
                coarse_preconditioner=NativeCoarsePreconditioner.from_operator(op,M)
                M=coarse_preconditioner.linear_operator
                preconditioner_metadata['coarse']=coarse_preconditioner.metadata
                preconditioner_metadata['setup_seconds']=perf_counter()-started
        k=1.;history=[];reason='outer_iteration_limit';numerical=False;full=float('inf')
        inner_warm_start_reuses=0
        def coarse_monitor():
            if accepted_outer_state_callback is None or preconditioner!='group_ilu_coarse':
                return {}
            metadata=coarse_preconditioner.metadata
            return dict(coarse_aggregate_sha256=metadata['aggregate_sha256'],
                coarse_aggregate_count=metadata['aggregate_count'],coarse_dimension=metadata['coarse_dimension'],
                coarse_applications=metadata['applications'],coarse_solves=metadata['coarse_solves'],
                coarse_extra_a_matvecs=metadata['extra_a_matvecs'],
                coarse_setup_coarse_solves=metadata['setup_coarse_solves'])
        # Keep the linear operator independent from the mutable assembled sparse
        # matrix so post-GMRES residual checks detect any assembly inconsistency.
        for outer in range(1,max_outer+1):
            rhs=op.fission_action(u);steps=[]
            reuse_inner=inner_warm_start and outer>1
            # Multiplication allocates a separate initial guess: a solver may
            # mutate x0 without changing the normalized outer candidate.
            inner_initial=k*u if reuse_inner else None
            if reuse_inner:inner_warm_start_reuses+=1
            warm_record=(dict(inner_warm_start=True,inner_initial_guess_reused=reuse_inner,
                             inner_initial_guess='k*u' if reuse_inner else 'zero',
                             restart=min(40,op.size)) if inner_warm_start else {})
            callback=steps.append
            if iteration_callback is not None:
                iteration_callback(dict(phase='inner_start',outer=outer,max_inner=max_inner,
                    restart=min(40,op.size),inner_tolerance=float(inner_tolerance),
                    **{key:value for key,value in warm_record.items() if key!='restart'},**coarse_monitor()))
                def callback(value):
                    steps.append(value)
                    iteration_callback(dict(phase='inner_iteration',outer=outer,inner_iteration=len(steps),
                        scipy_preconditioned_residual_norm=float(value),
                        residual_kind='scipy_preconditioned_relative_residual_monitor_only',used_for_acceptance=False))
            y,info=gmres(solve_matrix,rhs,x0=inner_initial,rtol=inner_tolerance,atol=0,M=M,
                         restart=min(40,op.size),maxiter=max_inner,callback=callback,callback_type='legacy')
            linear=float(np.linalg.norm(op.action(y)-rhs)/np.linalg.norm(rhs))
            record=dict(outer=outer,inner_iterations=len(steps),gmres_info=int(info),linear_residual=linear)
            record.update(warm_record)
            if iteration_callback is not None:
                iteration_callback(dict(phase='inner_complete',outer=outer,count=len(steps),gmres_info=int(info),
                    true_action_relative_residual=linear,accepted_linear=bool(info==0 and np.isfinite(linear) and linear<=inner_tolerance),
                    **warm_record,**coarse_monitor()))
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
            candidate_converged=k_error<=k_tolerance and source_error<=source_tolerance and full<=residual_tolerance
            if accepted_outer_state_callback is not None:
                accepted_outer_state_callback(dict(record,linear_solution=np.array(y,copy=True),
                    rhs=np.array(rhs,copy=True),native_state=np.array(u,copy=True),converged=bool(candidate_converged),
                    **coarse_monitor()))
            if candidate_converged:
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
        if preconditioner=='group_ilu_coarse':
            files += (Path(__file__).with_name('native_coarse.py'),)
            preconditioner_metadata['coarse']=coarse_preconditioner.metadata
        provenance=dict(physical_admission=False,computational_only=True,solver='CPU scipy GMRES + source-normalized power iteration',
            numpy_version=np.__version__,scipy_version=scipy.__version__,family=op.family,moments_per_cell=op.n_moments,
            required_orders=op.required_orders,material_provenance=[dict(m.provenance) for m in op.materials],
            implementation_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            max_inner=max_inner,max_outer=max_outer,inner_tolerance=inner_tolerance,k_tolerance=k_tolerance,
            source_tolerance=source_tolerance,residual_tolerance=residual_tolerance,positivity_tolerance=positivity_tolerance,
            group_ilu_fill_factor=group_fill if preconditioner in ('group_ilu','group_ilu_coarse') else None,
            source_normalization='sum(cell_volume * NSF dot reconstructed phi0) = 1',
            inner_warm_start=inner_warm_start,inner_initial_guess_reuse_count=inner_warm_start_reuses,
            initial_state_reused=False,regularization=False,fallback=False,scattering_RHS=False,
            preconditioner=preconditioner,preconditioner_metadata=preconditioner_metadata)
        if inner_warm_start:
            provenance['initial_guess_source']='previous source-normalized outer state scaled by k'
        return CoupledMomentResult(k,u.reshape(op.shape),phi,power,numerical and positive,numerical,positive,
                                   len(history),reason,full,tuple(history),provenance)
