"""Experimental CPU-only whole-face NEM diffusion receiver.

The retained unknowns are node means and one global-axis current density per
physical face. Loss-only local responses eliminate six moments and side traces;
no fission or reference auxiliary delta is hidden in that response. The explicit
one-sided transverse-leakage closure is supplied by CuboidNodalGrid. This is an
unqualified prototype, with no positivity theorem or full 3D point-flux claim.
"""
from dataclasses import dataclass, replace
from hashlib import sha256

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, onenormest, splu

from .nem_geometry import CuboidNodalGrid
from .nodal_polynomial_reference import _stationary_points

_PAIRS = ((0, 2), (1, 3), (4, 5))


@dataclass(frozen=True)
class NEMResult:
    means: np.ndarray
    moments: np.ndarray
    currents: np.ndarray
    side_face_means: np.ndarray
    corrected_face_means: np.ndarray
    diagnostics: dict
    k_eff: float | None = None
    iterations: int = 0


class NEMSolverError(RuntimeError):
    """Numerical rejection retaining the latest state and diagnostics."""

    def __init__(self, message, diagnostics, result=None):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.result = result


def _array(value, name):
    if hasattr(value, '__cuda_array_interface__') or hasattr(value, 'get'):
        raise TypeError(f'{name}: CPU arrays required')
    if np.iscomplexobj(value):
        raise ValueError(f'{name} must be real')
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be finite')
    return array


def _field(value, name, shape):
    array = _array(value, name)
    if array.shape not in (shape, shape[1:]):
        raise ValueError(f'{name} must have shape {shape} or {shape[1:]}')
    return np.broadcast_to(array, shape).copy()


def _relative(residual, scale):
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(scale)):
        return float('inf')
    return float(np.max(abs(residual)/np.maximum(scale, np.finfo(float).tiny)))


class NEMDiffusionSolver:
    """Cache the condensed loss matrix and sparse LU, using NumPy/SciPy only.

    D, sigma_a, nu_sigma_f and chi have shape (N,G), or (G,) to broadcast;
    source-row sigma_s has shape (N,G,G) or (G,G). Its diagonal is ignored:
    removal includes absorption plus OFF-diagonal outgoing scatter only.
    Positive node_face_factors use (N,6,G), sides x-,y-,x+,y+,z-,z+;
    factors on external faces must equal one. Interior continuity is
    f_low*b_low_plus=f_high*b_high_minus. Currents are positive in global axes.
    Only means/faces/1D profile minima are flagged; negative states are retained.
    Public data arrays are read-only copies. Each public solve rejects geometry
    or material attribute replacement after cached assembly. Local responses
    are gated by equilibrated dense 2-norm condition numbers; the global loss
    condition is an estimated 1-norm, not a certified upper bound.
    """

    def __init__(self, grid, D, sigma_a, sigma_s, nu_sigma_f, chi,
                 node_face_factors=None, *, rtol=1e-10, condition_limit=1e12):
        if not isinstance(grid, CuboidNodalGrid):
            raise TypeError('grid must be a CuboidNodalGrid')
        self.grid = grid
        self.n_nodes = N = grid.n_nodes
        self.n_faces = F = len(grid.face_nodes)
        raw = _array(D, 'D')
        if raw.ndim not in (1, 2) or not raw.shape[-1]:
            raise ValueError('D must be a nonempty group vector or node/group array')
        self.n_groups = G = raw.shape[-1]
        self.D = _field(raw, 'D', (N,G))
        self.sigma_a = _field(sigma_a, 'sigma_a', (N,G))
        self.sigma_s = _field(sigma_s, 'sigma_s', (N,G,G))
        self.nu_sigma_f = _field(nu_sigma_f, 'nu_sigma_f', (N,G))
        self.chi = _field(chi, 'chi', (N,G))
        if np.any(self.D <= 0) or any(np.any(x < 0) for x in (self.sigma_a,self.sigma_s,self.nu_sigma_f,self.chi)):
            raise ValueError('D must be positive; all cross sections and chi nonnegative')
        fuel = np.any(self.nu_sigma_f > 0, axis=1)
        if np.any(abs(self.chi.sum(axis=1)[fuel]-1) > 1e-12):
            raise ValueError('chi must sum to one in fission-producing nodes')
        if not np.isscalar(rtol) or not np.isfinite(rtol) or rtol <= 0:
            raise ValueError('rtol must be finite positive')
        if not np.isscalar(condition_limit) or not np.isfinite(condition_limit) or condition_limit < 1:
            raise ValueError('condition_limit must be finite and at least one')
        self.rtol = float(rtol)
        self.condition_limit = float(condition_limit)
        self.node_face_factors = (np.ones((N,6,G)) if node_face_factors is None
                                  else _field(node_face_factors,'node_face_factors',(N,6,G)))
        if np.any(self.node_face_factors <= 0):
            raise ValueError('node_face_factors must be positive')
        if np.any(self.node_face_factors[grid.boundary_nodes,grid.boundary_sides] != 1):
            raise ValueError('external node_face_factors must equal one')
        scatter = self.sigma_s.copy()
        scatter[:,np.arange(G),np.arange(G)] = 0
        self.R = -scatter.transpose(0,2,1)
        self.R[:,np.arange(G),np.arange(G)] += self.sigma_a+scatter.sum(axis=2)
        self.fission = self.chi[:,:,None]*self.nu_sigma_f[:,None,:]
        self._local_inverse = np.empty((N,3,2,G,G))
        local_condition = 0.
        for node in range(N):
            for axis in range(3):
                H = self.D[node]/grid.widths[node,axis]**2
                for order, shift in enumerate((10,42)):
                    block = self.R[node]+np.diag(shift*H)
                    if not np.all(np.isfinite(block)):
                        raise NEMSolverError('nonfinite local loss response',{})
                    scaled = block.copy()
                    row,column = np.ones(G),np.ones(G)
                    for _ in range(4):
                        maxima = abs(scaled).max(axis=1)
                        r = np.ones(G);r[maxima>0] = 1/maxima[maxima>0]
                        scaled *= r[:,None];row *= r
                        maxima = abs(scaled).max(axis=0)
                        c = np.ones(G);c[maxima>0] = 1/maxima[maxima>0]
                        scaled *= c[None,:];column *= c
                    try:
                        condition = float(np.linalg.cond(scaled,2))
                        if not np.isfinite(condition) or condition > self.condition_limit:
                            raise NEMSolverError('local loss response condition exceeds limit',
                                dict(node=node,axis=axis,order=order,condition_number_2_scaled=condition))
                        local_condition = max(local_condition,condition)
                        inverse = column[:,None]*np.linalg.solve(scaled,row[:,None]*np.eye(G))
                    except np.linalg.LinAlgError as exc:
                        raise NEMSolverError('singular local loss response',{}) from exc
                    if not np.all(np.isfinite(inverse)):
                        raise NEMSolverError('nonfinite local inverse response',{})
                    self._local_inverse[node,axis,order] = inverse
        self._assemble()
        self._factor(local_condition)
        for array in (self.D,self.sigma_a,self.sigma_s,self.nu_sigma_f,self.chi,
                      self.node_face_factors,self.R,self.fission,self._local_inverse):
            array.setflags(write=False)
        self._configuration_signature = self._signature()

    def _assemble(self):
        grid, G, N, F = self.grid,self.n_groups,self.n_nodes,self.n_faces
        mr,mc,mv = [],[],[]
        def mrow(node,axis,order,g):
            return ((node*3+axis)*2+order)*G+g
        for node in range(N):
            for axis,(sm,sp) in enumerate(_PAIRS):
                samples,weights = grid.leakage_weights(node,axis)
                for order in range(2):
                    coefficients = {}
                    def add(face,value):
                        coefficients[face] = coefficients.get(face,0.)+value
                    for sample_index,sample in enumerate(samples):
                        for transverse,(tm,tp) in enumerate(_PAIRS):
                            if transverse != axis:
                                w = weights[order,sample_index]/grid.widths[node,transverse]
                                add(grid.node_faces[sample,tp],-w)
                                add(grid.node_faces[sample,tm],w)
                    L = grid.widths[node,axis]
                    if order == 0:
                        add(grid.node_faces[node,sp],-5/(12*L))
                        add(grid.node_faces[node,sm],-5/(12*L))
                    else:
                        add(grid.node_faces[node,sp],-7/(20*L))
                        add(grid.node_faces[node,sm],7/(20*L))
                    inverse = self._local_inverse[node,axis,order]
                    for face,value in coefficients.items():
                        for g in range(G):
                            for j in range(G):
                                entry = inverse[g,j]*value
                                if entry != 0:
                                    mr.append(mrow(node,axis,order,g));mc.append(face*G+j);mv.append(entry)
        self._moment_current = sparse.coo_matrix((mv,(mr,mc)),shape=(6*N*G,F*G)).tocsr()
        tr,tc,tv,dr,dc,dv = [],[],[],[],[],[]
        for node in range(N):
            for axis,(sm,sp) in enumerate(_PAIRS):
                L = grid.widths[node,axis]
                fm,fp = grid.node_faces[node,[sm,sp]]
                for side,sign in ((sm,-1),(sp,1)):
                    for g in range(G):
                        row = (node*6+side)*G+g
                        tr.extend((row,row));tc.extend((mrow(node,axis,0,g),mrow(node,axis,1,g)));tv.extend((sign*5,7))
                        # -L/(40D)*(Jp-Jm) +/-[-L/(24D)*(Jp+Jm)]
                        dr.extend((row,row));dc.extend((fp*G+g,fm*G+g))
                        dv.extend((-L/(40*self.D[node,g])-sign*L/(24*self.D[node,g]),
                                   L/(40*self.D[node,g])-sign*L/(24*self.D[node,g])))
        self._face_from_moments = sparse.coo_matrix((tv,(tr,tc)),shape=(6*N*G,6*N*G)).tocsr()
        direct = sparse.coo_matrix((dv,(dr,dc)),shape=(6*N*G,F*G)).tocsr()
        self._face_current = self._face_from_moments@self._moment_current+direct
        mean_map = sparse.coo_matrix((np.ones(6*N*G),(np.arange(6*N*G),
            np.repeat(np.arange(N),6*G)*G+np.tile(np.arange(G),6*N))),shape=(6*N*G,N*G)).tocsr()
        self._face_mean = mean_map
        side_operator = sparse.hstack((mean_map,self._face_current),format='csr')
        wr,wc,wv = [],[],[]
        normal = np.zeros(F*G)
        boundary = {int(f):i for i,f in enumerate(grid.boundary_faces)}
        for face,(low,high) in enumerate(grid.face_nodes):
            axis = grid.face_axes[face]
            if low >= 0 and high >= 0:
                for node,side,sign in ((low,_PAIRS[axis][1],1),(high,_PAIRS[axis][0],-1)):
                    for g in range(G):
                        wr.append(face*G+g);wc.append((node*6+side)*G+g)
                        wv.append(sign*self.node_face_factors[node,side,g])
            else:
                b = boundary[face]
                node,side = grid.boundary_nodes[b],grid.boundary_sides[b]
                for g in range(G):
                    wr.append(face*G+g);wc.append((node*6+side)*G+g);wv.append(-grid.boundary_alpha[b])
                    normal[face*G+g] = grid.normal_signs[side]
        self._constraint_face = sparse.coo_matrix((wv,(wr,wc)),shape=(F*G,6*N*G)).tocsr()
        cr,cc,cv = [],[],[]
        for node in range(N):
            for axis,(sm,sp) in enumerate(_PAIRS):
                for side,sign in ((sm,-1),(sp,1)):
                    for g in range(G):
                        cr.append(node*G+g);cc.append(grid.node_faces[node,side]*G+g)
                        cv.append(sign/grid.widths[node,axis])
        self._divergence = sparse.coo_matrix((cv,(cr,cc)),shape=(N*G,F*G)).tocsr()
        top = sparse.hstack((sparse.block_diag([sparse.csr_matrix(block) for block in self.R],format='csr'),self._divergence))
        bottom = self._constraint_face@side_operator+sparse.hstack((sparse.csr_matrix((F*G,N*G)),sparse.diags(normal)))
        self._loss_matrix = sparse.vstack((top,bottom),format='csc')
        self._loss_matrix.eliminate_zeros()

    def _factor(self, local_condition):
        matrix = self._loss_matrix.copy()
        size = matrix.shape[0]
        row,column = np.ones(size),np.ones(size)
        for _ in range(4):
            maxima = abs(matrix).max(axis=1).toarray().ravel()
            r = np.ones(size);r[maxima>0] = 1/maxima[maxima>0]
            matrix = sparse.diags(r)@matrix;row *= r
            maxima = abs(matrix).max(axis=0).toarray().ravel()
            c = np.ones(size);c[maxima>0] = 1/maxima[maxima>0]
            matrix = (matrix@sparse.diags(c)).tocsc();column *= c
        self._row_scale,self._column_scale,self._scaled_matrix = row,column,matrix
        evidence = dict(backend='NumPy/SciPy CPU',unknowns=size,loss_matrix_nnz=int(matrix.nnz),
                        maximum_local_loss_condition_2_scaled=local_condition,
                        condition_estimate_is_certified_upper_bound=False)
        try:
            self._lu = splu(matrix,permc_spec='MMD_AT_PLUS_A')
            inverse = LinearOperator(matrix.shape,matvec=self._lu.solve,
                rmatvec=lambda x:self._lu.solve(x,trans='T'),matmat=self._lu.solve)
            condition = float(sparse.linalg.norm(matrix,1)*onenormest(inverse))
        except (RuntimeError,ValueError) as exc:
            raise NEMSolverError('singular condensed loss operator',evidence) from exc
        evidence.update(condition_number_1_scaled_estimate=condition,
                        linear_solver=type(self._lu).__name__,factor_nnz=int(self._lu.L.nnz+self._lu.U.nnz))
        if not np.isfinite(condition) or condition > self.condition_limit:
            raise NEMSolverError('condensed loss condition estimate exceeds limit',evidence)
        self._backend_evidence = evidence

    def _signature(self):
        """Guard even attribute replacement on the supplied read-only geometry."""
        digest = sha256()
        arrays = [self.D,self.sigma_a,self.sigma_s,self.nu_sigma_f,self.chi,
                  self.node_face_factors,self.R,self.fission]
        arrays += [getattr(self.grid,name) for name in (
            'bounds','widths','volumes','face_nodes','face_axes','face_areas',
            'node_faces','neighbors','boundary_faces','boundary_nodes',
            'boundary_sides','boundary_alpha','boundary_kinds','normal_signs')]
        for node in range(self.n_nodes):
            for axis in range(3):
                arrays.extend(self.grid.leakage_weights(node,axis))
        digest.update(str((self.grid.n_nodes,self.n_nodes,self.n_faces,self.n_groups,
                           self.rtol,self.condition_limit,self.grid.leakage_boundary)).encode())
        for array in arrays:
            digest.update(str((array.shape,array.dtype.str)).encode())
            digest.update(array.tobytes())
        return digest.digest()

    def _assert_unchanged(self):
        if self._signature() != self._configuration_signature:
            raise ValueError('NEM geometry or cross sections changed after cached loss assembly')

    def _sources(self,mean_source,moment_source,boundary_drive):
        N,G = self.n_nodes,self.n_groups
        q0 = _field(mean_source,'mean_source',(N,G))
        qm = (np.zeros((N,3,2,G)) if moment_source is None else
              _field(moment_source,'moment_source',(N,3,2,G)))
        B = len(self.grid.boundary_faces)
        drive = (np.zeros((B,G)) if boundary_drive is None else
                 _field(boundary_drive,'boundary_drive',(B,G)))
        return q0,qm,drive

    def _solve(self,q0,qm,drive,profiles):
        N,G,F = self.n_nodes,self.n_groups,self.n_faces
        moment_offset = np.einsum('napgh,naph->napg',self._local_inverse,qm)
        face_offset = self._face_from_moments@moment_offset.ravel()
        rhs_faces = -(self._constraint_face@face_offset).reshape(F,G)
        rhs_faces[self.grid.boundary_faces] += drive
        rhs = np.r_[q0.ravel(),rhs_faces.ravel()]
        try:
            solution_scaled = self._lu.solve(self._row_scale*rhs)
            solution = self._column_scale*solution_scaled
        except (RuntimeError,ValueError) as exc:
            raise NEMSolverError('condensed loss solve failed',self._backend_evidence.copy()) from exc
        means = solution[:N*G].reshape(N,G)
        currents = solution[N*G:].reshape(F,G)
        moments = (self._moment_current@currents.ravel()).reshape(N,3,2,G)+moment_offset
        faces = (self._face_mean@means.ravel()+self._face_current@currents.ravel()+face_offset).reshape(N,6,G)
        corrected = np.empty((F,G))
        for face,(low,high) in enumerate(self.grid.face_nodes):
            axis = self.grid.face_axes[face]
            node,side = (low,_PAIRS[axis][1]) if low >= 0 else (high,_PAIRS[axis][0])
            corrected[face] = self.node_face_factors[node,side]*faces[node,side]
        info = self._backend_evidence.copy()
        residual = self._scaled_matrix@solution_scaled-self._row_scale*rhs
        scale = abs(self._scaled_matrix)@abs(solution_scaled)+abs(self._row_scale*rhs)
        info['scaled_backward_error'] = _relative(residual,scale)
        result = NEMResult(means,moments,currents,faces,corrected,info)
        if not all(np.all(np.isfinite(x)) for x in (means,moments,currents,faces,corrected)):
            raise NEMSolverError('nonfinite NEM state',info,result)
        info.update(self._assess(result,q0,qm,drive,profiles))
        if any(not np.isfinite(info[k]) or info[k] > self.rtol for k in self._residual_keys):
            raise NEMSolverError('NEM physical residual exceeds tolerance',info,result)
        return result

    _residual_keys = ('volume_balance_error','weak_balance_error','current_balance_error',
                      'mean_balance_error','moment_balance_error','interface_balance_error','boundary_balance_error')

    def solve_fixed_source(self,mean_source,moment_source=None,boundary_drive=None):
        """Solve prescribed mean/moment sources and optional Robin drive[B,G].

        Moment source has shape (N,3,2,G). No spatial moment source is discarded.
        Negative means, faces or profiles are diagnosed, never clipped/rejected.
        """
        self._assert_unchanged()
        return self._solve(*self._sources(mean_source,moment_source,boundary_drive),profiles=True)

    def _leakage(self,currents):
        result = np.zeros((self.n_nodes,3,2,self.n_groups))
        grid = self.grid
        for node in range(self.n_nodes):
            for axis in range(3):
                samples,weights = grid.leakage_weights(node,axis)
                for transverse,(sm,sp) in enumerate(_PAIRS):
                    if transverse != axis:
                        difference = currents[grid.node_faces[samples,sp]]-currents[grid.node_faces[samples,sm]]
                        result[node,axis] += weights@difference/grid.widths[node,transverse]
        return result

    def _assess(self,result,q0,qm,drive,profiles):
        grid,G = self.grid,self.n_groups
        mu,mom,J,b = result.means,result.moments,result.currents,result.side_face_means
        leakage = self._leakage(J)
        divergence = np.stack([(J[grid.node_faces[:,sp]]-J[grid.node_faces[:,sm]])/grid.widths[:,a,None]
                               for a,(sm,sp) in enumerate(_PAIRS)],axis=1)
        volume_r = np.einsum('ngh,nh->ng',self.R,mu)+divergence.sum(axis=1)-q0
        volume_scale = np.einsum('ngh,nh->ng',abs(self.R),abs(mu))+abs(divergence).sum(axis=1)+abs(q0)
        t,w = np.polynomial.legendre.leggauss(8);t,w = t/2,w/2
        basis_at = np.array([np.ones_like(t),t,3*t*t-.25,t*(t*t-.25),(t*t-.05)*(t*t-.25)])
        second_at = np.array([np.zeros_like(t),np.zeros_like(t),np.full_like(t,6),6*t,12*t*t-.6])
        first_at = np.array([[0.,0.],[1.,1.],[-3.,3.],[.5,.5],[-.2,.2]])
        tests = np.array([t,3*t*t-.25])
        minima = np.empty((self.n_nodes,3,G));locations = np.empty_like(minima)
        weak_error=current_error=mean_error=moment_error=0.
        weak_raw = np.empty_like(mom)
        weak_scale_raw = np.empty_like(mom)
        current_raw = np.empty_like(mom)
        mean_raw = np.empty((self.n_nodes,3,G))
        moment_raw = np.empty_like(mom)
        for axis,(sm,sp) in enumerate(_PAIRS):
            a1,a2 = b[:,sp]-b[:,sm],b[:,sp]+b[:,sm]-2*mu
            a3,a4 = 10*a1-120*mom[:,axis,0],35*a2-700*mom[:,axis,1]
            coeff = np.stack((mu,a1,a2,a3,a4),axis=-1)
            values = coeff@basis_at
            second = coeff@second_at
            weighted = (values*w)@tests.T
            second_weighted = (second*w)@tests.T
            H = self.D/grid.widths[:,axis,None]**2
            wr = np.einsum('ngh,nhp->ngp',self.R,weighted)-H[:,:,None]*second_weighted+leakage[:,axis].transpose(0,2,1)-qm[:,axis].transpose(0,2,1)
            ws = (np.einsum('ngh,nhp->ngp',abs(self.R),(abs(values)*w)@abs(tests).T)
                  +H[:,:,None]*((abs(second)*w)@abs(tests).T)
                  +abs(leakage[:,axis].transpose(0,2,1))+abs(qm[:,axis].transpose(0,2,1))
                  +(H*abs(mu))[:,:,None]*(abs(tests)@w)[None,None,:])
            weak_raw[:,axis] = wr.transpose(0,2,1)
            weak_scale_raw[:,axis] = ws.transpose(0,2,1)
            weak_error = max(weak_error,_relative(wr,ws))
            endpoint_J = -self.D[:,:,None]/grid.widths[:,axis,None,None]*(coeff@first_at)
            prescribed = np.stack((J[grid.node_faces[:,sm]],J[grid.node_faces[:,sp]]),axis=-1)
            current_raw[:,axis] = (endpoint_J-prescribed).transpose(0,2,1)
            current_error = max(current_error,_relative(endpoint_J-prescribed,abs(endpoint_J)+abs(prescribed)+(self.D*abs(mu)/grid.widths[:,axis,None])[:,:,None]))
            flux_scale = np.max(abs(coeff),axis=-1)
            mean_raw[:,axis] = values@w-mu
            moment_raw[:,axis] = weighted.transpose(0,2,1)-mom[:,axis]
            mean_error = max(mean_error,_relative(values@w-mu,flux_scale))
            moment_error = max(moment_error,_relative(weighted-mom[:,axis].transpose(0,2,1),flux_scale[:,:,None]))
            if profiles:
                for node in range(self.n_nodes):
                    for g in range(G):
                        c = coeff[node,g]
                        power = np.array([c[0]-c[2]/4+c[4]/80,c[1]-c[3]/4,3*c[2]-.3*c[4],c[3],c[4]])
                        if not np.all(np.isfinite(power)):
                            raise NEMSolverError('nonfinite reconstructed profile',
                                dict(node=node,axis=axis,group=g),result)
                        scale = max(abs(power))
                        normalized = power/scale if scale else power
                        try:
                            points = np.r_[-.5,_stationary_points(normalized),.5]
                        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                            raise NEMSolverError('profile stationary-point calculation failed',
                                dict(node=node,axis=axis,group=g),result) from exc
                        values_at = np.polynomial.polynomial.polyval(points,normalized)*scale
                        index = np.argmin(values_at)
                        minima[node,axis,g],locations[node,axis,g] = values_at[index],points[index]
        interface_r,interface_s = [],[]
        for face,(low,high) in enumerate(grid.face_nodes):
            if low >= 0 and high >= 0:
                sm,sp = _PAIRS[grid.face_axes[face]]
                left = self.node_face_factors[low,sp]*b[low,sp]
                right = self.node_face_factors[high,sm]*b[high,sm]
                interface_r.extend(left-right);interface_s.extend(abs(left)+abs(right))
        boundary_b = b[grid.boundary_nodes,grid.boundary_sides]
        outward = grid.normal_signs[grid.boundary_sides,None]*J[grid.boundary_faces]
        robin = grid.boundary_alpha[:,None]*boundary_b
        info = dict(volume_balance_error=_relative(volume_r,volume_scale),weak_balance_error=weak_error,
                    current_balance_error=current_error,mean_balance_error=mean_error,moment_balance_error=moment_error,
                    interface_balance_error=_relative(np.asarray(interface_r),np.asarray(interface_s)) if interface_r else 0.,
                    boundary_balance_error=_relative(outward-robin-drive,abs(outward)+abs(robin)+abs(drive)+(self.D[grid.boundary_nodes]*abs(mu[grid.boundary_nodes])/grid.widths[grid.boundary_nodes,grid.face_axes[grid.boundary_faces],None])),
                    volume_residual=volume_r,volume_residual_scale=volume_scale,
                    weak_residual=weak_raw,weak_residual_scale=weak_scale_raw,
                    current_residual=current_raw,mean_residual=mean_raw,moment_residual=moment_raw,
                    interface_residual=np.asarray(interface_r).reshape(-1,G),
                    boundary_residual=outward-robin-drive,
                    weak_error_scale_definition='absolute Gauss reaction/diffusion contributions + abs(leakage) + abs(qmom) + (D/L^2)*abs(mean)*integral(abs(f_n))',
                    minimum_mean=float(np.min(mu)),minimum_side_face_mean=float(np.min(b)),
                    negative_mean_count=int(np.sum(mu<0)),negative_side_face_count=int(np.sum(b<0)))
        if profiles:
            info.update(profile_minima=minima,profile_minimum_locations=locations,
                        minimum_profile_flux=float(np.min(minima)),negative_profile_count=int(np.sum(minima<0)))
        return info

    def _production(self,result):
        return float(np.sum(self.grid.volumes[:,None]*self.nu_sigma_f*result.means))

    @staticmethod
    def _normalize(result,production):
        return replace(result,means=result.means/production,moments=result.moments/production,
                       currents=result.currents/production,side_face_means=result.side_face_means/production,
                       corrected_face_means=result.corrected_face_means/production)

    def solve_eigenvalue(self,*,rtol=1e-10,max_iterations=1000,initial_means=None,initial_moments=None):
        """Power iteration of A0^-1 F on ALL seven nodal flux moments.

        Boundary drive is exactly zero. Fission acts on the mean and every
        spatial moment, never face constraints. Production normalization scales
        the complete recovered state. Dominance of a positive physical mode is
        not guaranteed: competing/negative modes can prevent convergence.
        """
        self._assert_unchanged()
        if not np.isscalar(rtol) or not np.isfinite(rtol) or rtol <= 0:
            raise ValueError('rtol must be finite positive')
        if isinstance(max_iterations,(bool,np.bool_)) or not isinstance(max_iterations,(int,np.integer)) or max_iterations < 1:
            raise ValueError('max_iterations must be a positive integer')
        N,G,F = self.n_nodes,self.n_groups,self.n_faces
        means = np.ones((N,G)) if initial_means is None else _field(initial_means,'initial_means',(N,G))
        moments = np.zeros((N,3,2,G)) if initial_moments is None else _field(initial_moments,'initial_moments',(N,3,2,G))
        current = NEMResult(means,moments,np.zeros((F,G)),np.repeat(means[:,None,:],6,axis=1),np.zeros((F,G)),{})
        production = self._production(current)
        if not np.isfinite(production) or production <= 0:
            raise ValueError('eigen iteration needs positive finite initial fission production')
        current = self._normalize(current,production)
        drive = np.zeros((len(self.grid.boundary_faces),G))
        previous_k = None
        history = []
        for iteration in range(1,max_iterations+1):
            q0 = np.einsum('ngh,nh->ng',self.fission,current.means)
            qm = np.einsum('ngh,naph->napg',self.fission,current.moments)
            candidate = self._solve(q0,qm,drive,profiles=False)
            new_production = self._production(candidate)
            k = new_production/self._production(current)
            if not np.isfinite(k) or k <= 0 or new_production <= 0:
                candidate.diagnostics.update(self._assess(candidate,q0,qm,drive,profiles=True))
                raise NEMSolverError('nonpositive or nonfinite eigen production',candidate.diagnostics,candidate)
            candidate = self._normalize(candidate,new_production)
            target0 = np.einsum('ngh,nh->ng',self.fission,candidate.means)/k
            targetm = np.einsum('ngh,naph->napg',self.fission,candidate.moments)/k
            evidence = self._assess(candidate,target0,targetm,drive,profiles=False)
            error = max(evidence[key] for key in self._residual_keys)
            change = abs(k-previous_k)/abs(k) if previous_k is not None else float('inf')
            history.append(dict(iteration=iteration,k_eff=k,full_eigen_residual=error,relative_k_change=change if previous_k is not None else None))
            current = replace(candidate,k_eff=k,iterations=iteration)
            if error <= rtol and change <= rtol:
                info = self._backend_evidence.copy()
                info.update(self._assess(current,target0,targetm,drive,profiles=True))
                info.update(full_eigen_residual=error,power_iteration_history=history,
                            fission_production=self._production(current),all_spatial_fission_moments=True)
                return replace(current,diagnostics=info)
            previous_k = k
        info = current.diagnostics.copy()
        info.update(self._assess(current,target0,targetm,drive,profiles=True))
        info.update(full_eigen_residual=history[-1]['full_eigen_residual'],power_iteration_history=history)
        current = replace(current,diagnostics=info)
        raise NEMSolverError('NEM eigen iteration did not converge',info,current)
