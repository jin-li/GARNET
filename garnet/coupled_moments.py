"""Tiny CPU full-energy SP3/SDP1/SP5/SP7 orthogonal integrated TPFA operators.

Cell-major unknowns have shape (cells, moments, groups).
No eigenvalue driver, GPU, DF, regularization or physical admission is supplied.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from .moment_materials import MomentMaterial, _immutable
from .spn import _SPN_C, _SDPN_C, _SPN_G, _SDPN_G


@dataclass(frozen=True)
class InteriorFace:
    left: int
    right: int
    area: float
    center: tuple
    normal: tuple  # unit outward normal from left
    dleft: float
    dright: float


@dataclass(frozen=True)
class BoundaryFace:
    cell: int
    area: float
    center: tuple
    normal: tuple  # unit outward normal
    distance: float
    kind: str


@dataclass(frozen=True)
class OrthogonalMesh:
    volumes: np.ndarray
    centroids: np.ndarray
    interior: tuple = ()
    boundary: tuple = ()

    def __post_init__(self):
        object.__setattr__(self,'volumes',_immutable(np.asarray(self.volumes,dtype=float)))
        object.__setattr__(self,'centroids',_immutable(np.asarray(self.centroids,dtype=float)))
        object.__setattr__(self,'interior',tuple(self.interior))
        object.__setattr__(self,'boundary',tuple(self.boundary))


def _positive(value):
    return np.isfinite(value) and value>0


def _family_tables(family):
    """Bounded native families; no implicit higher-order fallback."""
    if family not in ('sp3', 'sdp1', 'sp5', 'sp7'):
        raise ValueError('Use sp3, sdp1, sp5 or sp7')
    order = 3 if family == 'sp7' else (2 if family == 'sp5' else 1)
    return (np.asarray((_SDPN_C if family == 'sdp1' else _SPN_C)[order], dtype=float),
            np.asarray((_SDPN_G if family == 'sdp1' else _SPN_G)[order], dtype=float))


def _geometry(mesh):
    v,c=mesh.volumes,mesh.centroids
    if v.ndim!=1 or not v.size or c.ndim!=2 or c.shape[0]!=v.size or c.shape[1] not in (1,2,3) or not np.all(np.isfinite(c)) or not np.all(np.isfinite(v)) or np.any(v<=0):
        raise ValueError('Invalid finite positive cell volumes/centroids')
    seen=set()
    def owner(i):
        if type(i) is not int or not 0<=i<len(v):raise ValueError('Invalid face cell owner')
    def plane(i,center,normal,distance):
        center,normal=np.asarray(center,dtype=float),np.asarray(normal,dtype=float)
        if center.shape!=c[i].shape or normal.shape!=c[i].shape or not np.all(np.isfinite(center)) or not np.all(np.isfinite(normal)) or not _positive(distance):raise ValueError('Invalid face geometry')
        if not np.isclose(np.linalg.norm(normal),1,rtol=1e-12,atol=1e-12) or not np.allclose(center-c[i],distance*normal,rtol=1e-12,atol=1e-12):raise ValueError('Orthogonal face alignment required')
    for face in mesh.interior:
        owner(face.left);owner(face.right)
        if face.left==face.right or not _positive(face.area):raise ValueError('Invalid interior face')
        plane(face.left,face.center,face.normal,face.dleft)
        plane(face.right,face.center,-np.asarray(face.normal),face.dright)
        key=('interior',tuple(sorted((face.left,face.right))),tuple(face.center))
        if key in seen:raise ValueError('Duplicate face')
        seen.add(key)
    for face in mesh.boundary:
        owner(face.cell)
        if face.kind not in ('reflective','vacuum') or not _positive(face.area):raise ValueError('Invalid boundary face')
        plane(face.cell,face.center,face.normal,face.distance)
        key=('boundary',face.cell,tuple(face.center))
        if key in seen:raise ValueError('Duplicate face')
        seen.add(key)


class CoupledMomentOperator:
    """Integrated nonsymmetric operator with transformed scalar reconstruction.

    All inversions reject finite condition numbers above condition_limit.
    Sparse assembly and matrix-free actions use the same frozen face blocks;
    neither adds a legacy off-group scattering RHS.
    """
    physical_admission=False

    def __init__(self,mesh,materials,*,family='sp3',condition_limit=1e12):
        if not isinstance(mesh,OrthogonalMesh):raise ValueError('Explicit OrthogonalMesh required')
        _geometry(mesh)
        coefficients, boundary_g = _family_tables(family)
        if not np.isfinite(condition_limit) or condition_limit<=1:raise ValueError('Invalid condition limit')
        materials=tuple(materials)
        if len(materials)!=len(mesh.volumes) or not materials or any(not isinstance(m,MomentMaterial) for m in materials):raise ValueError('One immutable MomentMaterial per cell required')
        self.n_groups=materials[0].n_groups;self.n_cells=len(materials);g=self.n_groups
        if any(m.n_groups!=g or m.provenance.get('orientation_contract') not in ('from-major','to-major') for m in materials):raise ValueError('Group/orientation identity differs')
        self.mesh=mesh;self.materials=materials;self.family=family
        self.n_moments=len(coefficients);self.required_orders=tuple(range(2*self.n_moments))
        self.shape=(self.n_cells,self.n_moments,g);self.size=self.n_cells*self.n_moments*g
        self.condition_limit=condition_limit
        self.C=tuple(_immutable(c) for c in coefficients)
        self.c1,self.c2=self.C[:2]
        self.B=_immutable(np.kron(boundary_g,np.eye(g)))
        reaction=[];diffusion=[]
        for v,m in zip(mesh.volumes,materials):
            if any(a.shape!=(g,) or not np.all(np.isfinite(a)) for a in (m.total,m.chi,m.nu_sigma_f,m.kappa_fission)):
                raise ValueError('Invalid finite native material vectors')
            if not np.array_equal(m.provenance.get('energy_bounds_mev'),materials[0].provenance.get('energy_bounds_mev')):
                raise ValueError('Energy boundary identity differs')
            if any(order not in m.loss for order in self.required_orders):raise ValueError(f'Native losses 0..{self.required_orders[-1]} required')
            for order in self.required_orders:
                if m.loss[order].shape!=(g,g) or not np.all(np.isfinite(m.loss[order])):raise ValueError('Invalid finite group loss matrix')
            reaction.append(v*sum(np.kron(c,m.loss[2*k]) for k,c in enumerate(self.C)))
            D=np.zeros((self.n_moments*g,self.n_moments*g))
            for k in range(self.n_moments):
                s=slice(k*g,(k+1)*g)
                D[s,s]=self._inverse(m.loss[2*k+1],f'Sigma{2*k+1}')/(4*k+3)
            diffusion.append(D)
            if not np.all(np.isfinite(reaction[-1])):
                raise ValueError('Nonfinite integrated reaction')
        self.reaction=tuple(_immutable(x) for x in reaction)
        self.diffusion=tuple(_immutable(x) for x in diffusion)
        faces=[];boundary=[]
        for f in mesh.interior:
            resistance=f.dleft*self._inverse(diffusion[f.left],'DL')+f.dright*self._inverse(diffusion[f.right],'DR')
            faces.append((f.left,f.right,_immutable(f.area*self._inverse(resistance,'face resistance'))))
        for f in mesh.boundary:
            if f.kind=='vacuum':
                C=diffusion[f.cell]/f.distance
                self._inverse(C+self.B,'vacuum C+B')
                boundary.append((f.cell,_immutable(f.area*self.B@np.linalg.solve(C+self.B,C))))
        if any(not np.all(np.isfinite(a)) for _,_,a in faces) or any(not np.all(np.isfinite(a)) for _,a in boundary):
            raise ValueError('Nonfinite integrated face block')
        self.faces=tuple(faces);self.boundary=tuple(boundary)
        rows=[];cols=[];data=[];q=self.n_moments*g
        def block(i,j,a):
            rr,cc=np.indices(a.shape)
            rows.extend((i*q+rr.ravel()).tolist());cols.extend((j*q+cc.ravel()).tolist());data.extend(a.ravel().tolist())
        for i,a in enumerate(self.reaction):block(i,i,a)
        for i,j,t in self.faces:
            block(i,i,t);block(j,j,t);block(i,j,-t);block(j,i,-t)
        for i,a in self.boundary:block(i,i,a)
        self.assembled=sparse.coo_matrix((data,(rows,cols)),shape=(self.size,self.size)).tocsr()

    def _inverse(self,m,label):
        m=np.asarray(m,dtype=float)
        if m.ndim!=2 or m.shape[0]!=m.shape[1] or not np.all(np.isfinite(m)):raise ValueError('Invalid '+label)
        condition=np.linalg.cond(m)
        if not np.isfinite(condition) or condition>self.condition_limit:raise ValueError('Singular/ill-conditioned '+label)
        return np.linalg.solve(m,np.eye(m.shape[0]))

    def _values(self,u):
        u=np.asarray(u,dtype=float)
        if u.shape not in (self.shape,(self.size,)) or not np.all(np.isfinite(u)):raise ValueError('Invalid finite moment unknowns')
        return u.reshape(self.n_cells,self.n_moments*self.n_groups)

    def action(self,u):
        x=self._values(u);out=np.stack([a@v for a,v in zip(self.reaction,x)])
        for i,j,t in self.faces:
            current=t@(x[i]-x[j]);out[i]+=current;out[j]-=current
        for i,t in self.boundary:out[i]+=t@x[i]
        return out.reshape(np.shape(u))

    def scalar_flux(self,u):
        x=self._values(u).reshape(self.shape)
        scalar=x[:,0,:]-2*x[:,1,:]/3
        if self.n_moments>=3:scalar=scalar+8*x[:,2,:]/15
        if self.n_moments==4:scalar=scalar-16*x[:,3,:]/35
        return scalar

    def fission_action(self,u):
        phi=self.scalar_flux(u)
        result=np.stack([v*self.c1[:,0,None]*m.chi[None,:]*np.dot(m.nu_sigma_f,p)
                         for v,m,p in zip(self.mesh.volumes,self.materials,phi)])
        return result.reshape(np.shape(u))
