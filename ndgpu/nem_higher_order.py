"""Matching CPU-only higher-order NEM receiver; unqualified research prototype.

Native whole faces and the quadratic transverse leakage prescription are
retained. Local coefficients are eliminated from the global system. Fission
acts on every spatial moment through the inherited complete-state iteration.
"""
import numpy as np
from scipy import sparse
from .nem import NEMDiffusionSolver,NEMResult,NEMSolverError,_field,_relative,_PAIRS
from .nem_polynomial_axis import NEMPolynomialAxisResponse,profile_minimum


class NEMHigherOrderDiffusionSolver(NEMDiffusionSolver):
    """Legendre NEM degree4/6/8/10/12/14; mean/J retained, P-2 moments per axis.

    Default degree14 is supported by focal-reference convergence only; this
    does not qualify a full-core calculation. All boundary policies remain
    explicit properties of CuboidNodalGrid. Negative physical states survive.
    """
    def __init__(self,*args,degree=14,**kwargs):
        if isinstance(degree,(bool,np.bool_)) or degree not in (4,6,8,10,12,14):
            raise ValueError('degree must be one of 4,6,8,10,12,14')
        self.degree=int(degree);self.n_moments=self.degree-2
        super().__init__(*args,**kwargs)

    def _signature(self):
        return super()._signature()+str(self.degree).encode()

    def _assemble(self):
        grid,G,N,F,P,K=self.grid,self.n_groups,self.n_nodes,self.n_faces,self.degree,self.n_moments
        cr,cc,cv=[],[],[]
        sr,sc,sv=[],[],[]
        self._higher_local_condition=0.
        for node in range(N):
            for axis,(sm,sp) in enumerate(_PAIRS):
                response=NEMPolynomialAxisResponse(self.D[node],self.R[node],grid.widths[node,axis],P,self.condition_limit)
                self._higher_local_condition=max(self._higher_local_condition,response.condition)
                # RHS rows first K are qmom - ellmom; final two are J-/J+.
                rhs_current={}
                def add(face,row,value):
                    key=(int(face),int(row));rhs_current[key]=rhs_current.get(key,0.)+value
                samples,weights=grid.leakage_weights(node,axis)
                for order in range(2):
                    for k,sample in enumerate(samples):
                        for transverse,(tm,tp) in enumerate(_PAIRS):
                            if transverse!=axis:
                                value=weights[order,k]/grid.widths[node,transverse]
                                add(grid.node_faces[sample,tp],order,-value)
                                add(grid.node_faces[sample,tm],order,value)
                add(grid.node_faces[node,sm],K,1.)
                add(grid.node_faces[node,sp],K+1,1.)
                row0=(node*3+axis)*P*G
                for (face,r),value in rhs_current.items():
                    block=response.response[:,r*G:(r+1)*G]*value
                    ii,jj=np.nonzero(block)
                    cr.extend(row0+ii);cc.extend(face*G+jj);cv.extend(block[ii,jj])
                block=response.response[:,:K*G]
                ii,jj=np.nonzero(block)
                sr.extend(row0+ii);sc.extend((node*3+axis)*K*G+jj);sv.extend(block[ii,jj])
        self._coefficient_current=sparse.coo_matrix((cv,(cr,cc)),shape=(N*3*P*G,F*G)).tocsr()
        self._coefficient_source=sparse.coo_matrix((sv,(sr,sc)),shape=(N*3*P*G,N*3*K*G)).tocsr()
        er,ec,ev=[],[],[];mr,mc,mv=[],[],[]
        for node in range(N):
            for axis,(sm,sp) in enumerate(_PAIRS):
                for j in range(1,P+1):
                    for g in range(G):
                        col=((node*3+axis)*P+j-1)*G+g
                        for side,value in ((sm,(-1)**j),(sp,1)):
                            er.append((node*6+side)*G+g);ec.append(col);ev.append(value)
                        if j<=K:
                            mr.append(((node*3+axis)*K+j-1)*G+g);mc.append(col);mv.append(1/(2*(2*j+1)))
        self._endpoint_coeff=sparse.coo_matrix((ev,(er,ec)),shape=(6*N*G,3*N*P*G)).tocsr()
        self._moment_coeff=sparse.coo_matrix((mv,(mr,mc)),shape=(3*N*K*G,3*N*P*G)).tocsr()
        self._moment_current=self._moment_coeff@self._coefficient_current
        self._moment_source=self._moment_coeff@self._coefficient_source
        self._face_current=self._endpoint_coeff@self._coefficient_current
        self._face_source=self._endpoint_coeff@self._coefficient_source
        self._assemble_constraints()

    def _factor(self,local_condition):
        super()._factor(max(local_condition,self._higher_local_condition))
        self._backend_evidence.update(polynomial_degree=self.degree,spatial_moments_per_axis=self.n_moments)

    def solve_fixed_source(self,mean_source,moment_source=None,boundary_drive=None):
        """Mean source N,G; signed spatial source N,3,P-2,G; Robin drive B,G."""
        return super().solve_fixed_source(mean_source,moment_source,boundary_drive)

    def solve_eigenvalue(self,**kwargs):
        """Complete-state power iteration, fission on all 1+3(P-2) moments."""
        return super().solve_eigenvalue(**kwargs)

    def _sources(self,mean_source,moment_source,boundary_drive):
        N,G,K=self.n_nodes,self.n_groups,self.n_moments
        q0=_field(mean_source,'mean_source',(N,G))
        qm=np.zeros((N,3,K,G)) if moment_source is None else _field(moment_source,'moment_source',(N,3,K,G))
        B=len(self.grid.boundary_faces)
        drive=np.zeros((B,G)) if boundary_drive is None else _field(boundary_drive,'boundary_drive',(B,G))
        return q0,qm,drive

    def _solve(self,q0,qm,drive,profiles):
        N,G,F,K=self.n_nodes,self.n_groups,self.n_faces,self.n_moments
        face_offset=self._face_source@qm.ravel()
        rhs_faces=-(self._constraint_face@face_offset).reshape(F,G)
        rhs_faces[self.grid.boundary_faces]+=drive
        rhs=np.r_[q0.ravel(),rhs_faces.ravel()]
        solution_scaled=self._lu.solve(self._row_scale*rhs)
        solution=self._column_scale*solution_scaled
        means=solution[:N*G].reshape(N,G);currents=solution[N*G:].reshape(F,G)
        moments=(self._moment_current@currents.ravel()+self._moment_source@qm.ravel()).reshape(N,3,K,G)
        faces=(self._face_mean@means.ravel()+self._face_current@currents.ravel()+face_offset).reshape(N,6,G)
        corrected=np.empty((F,G))
        for face,(low,high) in enumerate(self.grid.face_nodes):
            axis=self.grid.face_axes[face]
            node,side=(low,_PAIRS[axis][1]) if low>=0 else (high,_PAIRS[axis][0])
            corrected[face]=self.node_face_factors[node,side]*faces[node,side]
        info=self._backend_evidence.copy()
        info['scaled_backward_error']=_relative(self._scaled_matrix@solution_scaled-self._row_scale*rhs,abs(self._scaled_matrix)@abs(solution_scaled)+abs(self._row_scale*rhs))
        result=NEMResult(means,moments,currents,faces,corrected,info)
        if not all(np.all(np.isfinite(x)) for x in (means,moments,currents,faces,corrected)):
            raise NEMSolverError('nonfinite higher-order NEM state',info,result)
        info.update(self._assess(result,q0,qm,drive,profiles))
        if any(not np.isfinite(info[k]) or info[k]>self.rtol for k in self._residual_keys):
            raise NEMSolverError('NEM physical residual exceeds tolerance',info,result)
        return result

    def _leakage(self,currents):
        old=super()._leakage(currents)
        result=np.zeros((self.n_nodes,3,self.n_moments,self.n_groups));result[:,:,:2]=old
        return result

    def _assess(self,result,q0,qm,drive,profiles):
        grid,G,N,P,K=self.grid,self.n_groups,self.n_nodes,self.degree,self.n_moments
        mu,mom,J,b=result.means,result.moments,result.currents,result.side_face_means
        ell=self._leakage(J)
        div=np.stack([(J[grid.node_faces[:,sp]]-J[grid.node_faces[:,sm]])/grid.widths[:,a,None] for a,(sm,sp) in enumerate(_PAIRS)],axis=1)
        vr=np.einsum('ngh,nh->ng',self.R,mu)+div.sum(axis=1)-q0
        vs=np.einsum('ngh,nh->ng',abs(self.R),abs(mu))+abs(div).sum(axis=1)+abs(q0)
        leg=np.polynomial.legendre;u,w=leg.leggauss(P+2);w=w/2
        basis=leg.legvander(u,P).T;tests=basis[1:K+1]/2
        second=np.stack([leg.legval(u,leg.legder(np.eye(P+1)[j],2))*4 for j in range(P+1)])
        first=np.stack([leg.legval([-1.,1.],leg.legder(np.eye(P+1)[j]))*2 for j in range(P+1)])
        wr=np.empty_like(mom);ws=np.empty_like(mom);mr=np.empty_like(mom)
        current_raw=np.empty((N,3,2,G));mean_raw=np.empty((N,3,G))
        minima=np.empty((N,3,G));locations=np.empty_like(minima)
        weak_error=current_error=mean_error=moment_error=0.
        for axis,(sm,sp) in enumerate(_PAIRS):
            coeff=np.zeros((N,G,P+1));coeff[:,:,0]=mu
            coeff[:,:,1:K+1]=mom[:,axis].transpose(0,2,1)*(2*(2*np.arange(1,K+1)+1))
            coeff[:,:,P-1]=(b[:,sp]-b[:,sm])/2-coeff[:,:,1:K+1:2].sum(axis=2)
            coeff[:,:,P]=(b[:,sp]+b[:,sm])/2-mu-coeff[:,:,2:K+1:2].sum(axis=2)
            values=coeff@basis;dd=coeff@second;weighted=(values*w)@tests.T
            H=self.D/grid.widths[:,axis,None]**2
            weak=np.einsum('ngh,nhp->ngp',self.R,weighted)-H[:,:,None]*((dd*w)@tests.T)+ell[:,axis].transpose(0,2,1)-qm[:,axis].transpose(0,2,1)
            scale=(np.einsum('ngh,nhp->ngp',abs(self.R),(abs(values)*w)@abs(tests).T)+H[:,:,None]*((abs(dd)*w)@abs(tests).T)+abs(ell[:,axis].transpose(0,2,1))+abs(qm[:,axis].transpose(0,2,1))+(H*abs(mu))[:,:,None]*(abs(tests)@w))
            wr[:,axis]=weak.transpose(0,2,1);ws[:,axis]=scale.transpose(0,2,1)
            weak_error=max(weak_error,_relative(weak,scale))
            endpoint=-self.D[:,:,None]/grid.widths[:,axis,None,None]*(coeff@first)
            prescribed=np.stack((J[grid.node_faces[:,sm]],J[grid.node_faces[:,sp]]),axis=-1)
            current_raw[:,axis]=(endpoint-prescribed).transpose(0,2,1)
            current_error=max(current_error,_relative(endpoint-prescribed,abs(endpoint)+abs(prescribed)+(self.D*abs(mu)/grid.widths[:,axis,None])[:,:,None]))
            mean_raw[:,axis]=values@w-mu;mr[:,axis]=weighted.transpose(0,2,1)-mom[:,axis]
            flux_scale=np.max(abs(coeff),axis=2)
            mean_error=max(mean_error,_relative(mean_raw[:,axis],flux_scale))
            moment_error=max(moment_error,_relative(mr[:,axis].transpose(0,2,1),flux_scale[:,:,None]))
            if profiles:
                for node in range(N):
                    for g in range(G):minima[node,axis,g],locations[node,axis,g]=profile_minimum(coeff[node,g])
        ir,iscale=[],[]
        for face,(low,high) in enumerate(grid.face_nodes):
            if low>=0 and high>=0:
                sm,sp=_PAIRS[grid.face_axes[face]]
                left=self.node_face_factors[low,sp]*b[low,sp];right=self.node_face_factors[high,sm]*b[high,sm]
                ir.extend(left-right);iscale.extend(abs(left)+abs(right))
        boundary_b=b[grid.boundary_nodes,grid.boundary_sides]
        outward=grid.normal_signs[grid.boundary_sides,None]*J[grid.boundary_faces]
        robin=grid.boundary_alpha[:,None]*boundary_b
        info=dict(volume_balance_error=_relative(vr,vs),weak_balance_error=weak_error,current_balance_error=current_error,mean_balance_error=mean_error,moment_balance_error=moment_error,
                  interface_balance_error=_relative(np.array(ir),np.array(iscale)) if ir else 0.,
                  boundary_balance_error=_relative(outward-robin-drive,abs(outward)+abs(robin)+abs(drive)+(self.D[grid.boundary_nodes]*abs(mu[grid.boundary_nodes])/grid.widths[grid.boundary_nodes,grid.face_axes[grid.boundary_faces],None])),
                  volume_residual=vr,volume_residual_scale=vs,weak_residual=wr,weak_residual_scale=ws,current_residual=current_raw,mean_residual=mean_raw,moment_residual=mr,interface_residual=np.asarray(ir).reshape(-1,G),boundary_residual=outward-robin-drive,
                  minimum_mean=float(np.min(mu)),minimum_side_face_mean=float(np.min(b)),negative_mean_count=int(np.sum(mu<0)),negative_side_face_count=int(np.sum(b<0)),extrema_are_interval_certified=False)
        if profiles:info.update(profile_minima=minima,profile_minimum_locations=locations,minimum_profile_flux=float(np.min(minima)),negative_profile_count=int(np.sum(minima<0)))
        return info
