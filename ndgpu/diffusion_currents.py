"""Host export of ADF-off Cartesian diffusion TPFA face currents.

This module observes a solver's existing scalar state and operators; it never
builds or changes an operator or runs a solve. Currents are integrated and
outward from the first owner. Sides are W,S,E,N,bottom,top. Vertex-free mesh
geometry requires explicit cuboid widths and face sides (e.g. metadata from
``build_rectilinear_mesh_3d``); no centroid midpoint is substituted for a face.
"""
from __future__ import annotations

import numpy as np

from .backend import asnumpy
from .mesh import MeshGrid
from .solver import DiffusionEigenSolver
from .stencil import face_alpha, normalize_bc

_NORMALS = np.asarray(((-1,0,0),(0,-1,0),(1,0,0),(0,1,0),(0,0,-1),(0,0,1)))
_MINUS, _PLUS = (0,1,4), (2,3,5)


def _finite(value, shape, name, positive=False):
    result = np.asarray(asnumpy(value), dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)) or (positive and np.any(result <= 0)):
        raise ValueError(f'{name} must have finite {"positive " if positive else ""}shape {shape}')
    return result


def _indices(value, shape, upper, name):
    values = _finite(value, shape, name)
    if np.any(values != np.floor(values)) or np.any(values < 0) or np.any(values >= upper):
        raise ValueError(f'{name} contains invalid integer indices')
    return values.astype(np.int64)


def _structured_geometry(grid, bc):
    if bc is None:
        raise ValueError('Structured boundary_conditions must be explicit; operators do not retain their boundary spec')
    shape = tuple(grid.shape)
    spacing = np.asarray(grid.spacing, dtype=float)
    centers = np.stack(np.meshgrid(*(grid.cell_centers(a) for a in range(3)), indexing='ij'), axis=-1).reshape(-1,3)
    ids = np.arange(np.prod(shape)).reshape(shape)
    faces, sides, boundaries, bsides, specs = [], [], [], [], []
    conditions = normalize_bc(bc)
    for axis in range(3):
        measure = float(np.prod(np.delete(spacing,axis)))
        lo, hi = [slice(None)]*3, [slice(None)]*3
        lo[axis], hi[axis] = slice(None,-1), slice(1,None)
        for i,j in zip(ids[tuple(lo)].ravel(),ids[tuple(hi)].ravel()):
            faces.append((i,j,measure,spacing[axis]))
            sides.append((_PLUS[axis],_MINUS[axis]))
        for end,side in enumerate((_MINUS[axis],_PLUS[axis])):
            sl=[slice(None)]*3;sl[axis]=0 if end==0 else -1
            for i in ids[tuple(sl)].ravel():
                boundaries.append((i,measure,spacing[axis]/2))
                bsides.append(side);specs.append(conditions[axis][end])
    return dict(centers=centers,widths=np.tile(spacing,(len(centers),1)),
        volumes=np.full(len(centers),np.prod(spacing)),faces=np.asarray(faces).reshape(-1,4),
        boundaries=np.asarray(boundaries).reshape(-1,3),sides=np.asarray(sides).reshape(-1,2),
        bsides=np.asarray(bsides),distances=np.asarray([[f[3]/2]*2 for f in faces]).reshape(-1,2),
        specs=specs,operator_form='density',legacy=False)


def _mesh_geometry(grid, geometry, bc):
    if not isinstance(geometry,dict):
        raise ValueError('Mesh geometry requires widths, face_sides and boundary_sides metadata')
    mesh=grid.mesh;n=mesh.n_cells
    specs=grid.classify_boundary()
    if bc is not None and grid.classify_boundary(bc)!=specs:
        raise ValueError('Boundary override differs from authoritative mesh boundary conditions')
    return dict(centers=mesh.centroid,widths=geometry['widths'],volumes=mesh.area,
        faces=np.asarray(mesh.faces).reshape(-1,4),boundaries=np.asarray(mesh.bfaces).reshape(-1,3),
        sides=geometry['face_sides'],bsides=geometry['boundary_sides'],
        distances=(np.asarray(mesh.face_distances) if mesh.face_distances is not None else
                   np.asarray([[f[3]/2]*2 for f in mesh.faces]).reshape(-1,2)),
        specs=specs,operator_form='integrated',legacy=mesh.face_distances is None)


def _validate_geometry(data, n, rtol):
    centers=_finite(data['centers'],(n,3),'cell centroids')
    widths=_finite(data['widths'],(n,3),'cuboid widths',True)
    volumes=_finite(data['volumes'],(n,),'cell volumes',True)
    faces=np.asarray(data['faces']);boundaries=np.asarray(data['boundaries'])
    nf,nb=len(faces),len(boundaries)
    faces=_finite(faces,(nf,4),'interior faces');boundaries=_finite(boundaries,(nb,3),'boundary faces')
    owners=_indices(faces[:,:2],(nf,2),n,'interior owners')
    bowners=_indices(boundaries[:,0],(nb,),n,'boundary owners')
    sides=_indices(data['sides'],(nf,2),6,'interior sides')
    bsides=_indices(data['bsides'],(nb,),6,'boundary sides')
    distances=_finite(data['distances'],(nf,2),'normal distances',True)
    areas=_finite(faces[:,2],(nf,),'interior areas',True)
    _finite(faces[:,3],(nf,),'centroid distances',True)
    bareas=_finite(boundaries[:,1],(nb,),'boundary areas',True)
    bd=_finite(boundaries[:,2],(nb,),'boundary normal distances',True)
    if np.any(owners[:,0]==owners[:,1]) or np.any(_NORMALS[sides[:,0]] != -_NORMALS[sides[:,1]]):
        raise ValueError('Interior owners must differ and face normals must oppose')
    axes=np.argmax(abs(_NORMALS[sides[:,0]]),axis=1)
    baxes=np.argmax(abs(_NORMALS[bsides]),axis=1)
    def close(a,b,name):
        if not np.allclose(a,b,rtol=rtol,atol=rtol):
            raise ValueError('Cuboid geometry mismatch: '+name)
    close(volumes,widths.prod(axis=1),'volume')
    close(distances,widths[owners,axes[:,None]]/2,'interior half widths')
    close(bd,widths[bowners,baxes]/2,'boundary half widths')
    first=centers[owners[:,0]]+_NORMALS[sides[:,0]]*distances[:,0,None]
    second=centers[owners[:,1]]+_NORMALS[sides[:,1]]*distances[:,1,None]
    close(first,second,'owner/neighbor face centers (skew/hanging faces unsupported)')
    close(np.linalg.norm(centers[owners[:,0]]-centers[owners[:,1]],axis=1),faces[:,3],'centroid separation')
    expected=widths.prod(axis=1)[:,None]/widths
    close(areas,expected[owners[:,0],axes],'owner face area')
    close(areas,expected[owners[:,1],axes],'neighbor face area')
    close(bareas,expected[bowners,baxes],'boundary face area')
    # A full conforming cuboid has one observation of each of its six sides.
    counts=np.zeros((n,6),dtype=int);coverage=np.zeros((n,6))
    for end in range(2):
        np.add.at(counts,(owners[:,end],sides[:,end]),1)
        np.add.at(coverage,(owners[:,end],sides[:,end]),areas)
    np.add.at(counts,(bowners,bsides),1);np.add.at(coverage,(bowners,bsides),bareas)
    if np.any(counts!=1):
        raise ValueError('Cuboid face coverage must contain each owner side exactly once')
    close(coverage,expected[:,[0,1,0,1,2,2]],'six-side area coverage')
    if len(data['specs'])!=nb:
        raise ValueError('Boundary condition inventory differs')
    alpha=np.asarray([face_alpha(s) for s in data['specs']],dtype=float)
    if np.any(np.isnan(alpha)) or np.any(alpha<0):
        raise ValueError('Boundary coefficients must be nonnegative and not NaN')
    return dict(owners=owners,sides=sides,areas=areas,centers=first,normal_distances=distances,
        bowners=bowners,bsides=bsides, bareas=bareas,bd=bd,
        bcenters=centers[bowners]+_NORMALS[bsides]*bd[:,None],alpha=alpha,
        volumes=volumes,area_coverage=coverage)


def export_diffusion_face_currents(solver, *, flux=None, geometry=None,
                                   boundary_conditions=None, rtol=1e-11):
    """Export actual scalar diffusion state currents and verify operator identity.

    Returns a dict of host NumPy arrays. ``interior`` has owner/neighbor IDs and
    sides, area, true face center, normal distances, integrated outward owner
    current (G,F), its exact negative for the neighbor, and current densities.
    ``boundary`` has owner/side, area/center, alpha, outward integrated current
    and density. ``cell_outgoing`` and ``integrated_loss`` are (G,N).

    Requires ``solver.state``; optional ``flux`` must exactly equal that state
    (e.g. result.flux; flattened (G,N) is also accepted). Supports full uniform Cartesian grids and conforming
    orthogonal 3D cuboid MeshGrid only, with every cell side covered once.
    Skew/hanging faces, active masks, moment models and nonunity ADF are refused.
    Mesh metadata is explicit; structured boundary conditions are explicit and
    checked by the actual operator comparison. Finite signed flux is allowed.
    Operator checking uses max(rtol,64*eps(solver.dtype)), reported explicitly.
    No convergence, eigenbalance or physical positivity claim is made.
    """
    if not isinstance(solver,DiffusionEigenSolver):
        raise ValueError('Only scalar diffusion eigen solvers are supported')
    if not np.isfinite(rtol) or not 0<rtol<1:
        raise ValueError('rtol must be finite and between zero and one')
    if getattr(solver,'active',None) is not None or getattr(solver,'moment_df_model',None) is not None:
        raise ValueError('Active masks and moment models are unsupported')
    for name in ('face_df','boundary_df','df','bcf','moment_face_df','moment_boundary_df'):
        value=getattr(solver,name,None)
        if value is not None and not np.all(np.asarray(asnumpy(value))==1):
            raise ValueError('Face current export requires ADF off (exact unity factors)')
    if not hasattr(solver,'state'):
        raise ValueError('An existing scalar solver.state is required')
    grid=solver.grid;shape=tuple(grid.shape);groups=solver.n_groups;n=int(np.prod(shape))
    state=np.asarray([asnumpy(s) for s in solver.state])
    state=_finite(state,(groups,*shape),'scalar solver state')
    if flux is not None:
        supplied=np.asarray(asnumpy(flux))
        if supplied.shape==(groups,n):supplied=supplied.reshape((groups,*shape))
        supplied=_finite(supplied,(groups,*shape),'supplied flux')
        if not np.array_equal(supplied,state):
            raise ValueError('Supplied flux differs from actual solver.state')
    if isinstance(grid,MeshGrid):
        data=_mesh_geometry(grid,geometry,boundary_conditions)
    elif getattr(grid,'geometry',None)=='cartesian':
        if geometry is not None:
            raise ValueError('Structured geometry is inferred; metadata override is unsupported')
        data=_structured_geometry(grid,boundary_conditions)
    else:
        raise ValueError('Only Cartesian/cuboid geometry is supported')
    geo=_validate_geometry(data,n,rtol)
    phi=state.reshape(groups,n)
    D=np.asarray([asnumpy(d).reshape(-1) for d in solver.fields.diffusion])
    removal=np.asarray([asnumpy(r).reshape(-1) for r in solver.fields.removal])
    D=_finite(D,(groups,n),'diffusion coefficients',True)
    removal=_finite(removal,(groups,n),'removal coefficients')
    if np.any(removal<0):raise ValueError('Removal coefficients must be nonnegative')
    owners=geo['owners'];i,j=owners.T
    areas,dist=geo['areas'],geo['normal_distances']
    resistance=dist[:,0]/D[:,i]+dist[:,1]/D[:,j]
    weights=areas/resistance
    if data['legacy']:
        # Match the legacy MeshGroupOperator arithmetic as well as its law.
        distance=np.asarray(data['faces'])[:,3]
        weights=areas*2*D[:,i]*D[:,j]/((D[:,i]+D[:,j])*distance)
    integrated=weights*(phi[:,i]-phi[:,j])
    b=geo['bowners'];alpha=geo['alpha'];bd=geo['bd'];bareas=geo['bareas']
    bweights=np.empty((groups,len(b)))
    finite=np.isfinite(alpha)
    bweights[:,finite]=bareas[finite]*alpha[finite]*D[:,b[finite]]/(bd[finite]*alpha[finite]+D[:,b[finite]])
    bweights[:,~finite]=bareas[~finite]*D[:,b[~finite]]/bd[~finite]
    boundary=phi[:,b]*bweights
    if not all(np.all(np.isfinite(a)) for a in (weights,integrated,bweights,boundary)):
        raise ValueError('Nonfinite computed currents or coefficients')
    if np.any(weights<=0) or np.any(bweights<0):
        raise ValueError('Invalid face conductance')
    outgoing=np.zeros((groups,n));gross=np.zeros_like(outgoing)
    for g in range(groups):
        np.add.at(outgoing[g],i,integrated[g]);np.add.at(outgoing[g],j,-integrated[g])
        np.add.at(outgoing[g],b,boundary[g])
        np.add.at(gross[g],i,abs(integrated[g]));np.add.at(gross[g],j,abs(integrated[g]))
        np.add.at(gross[g],b,abs(boundary[g]))
    reaction=removal*phi*geo['volumes']
    loss=outgoing+reaction
    if len(solver.ops)!=groups:raise ValueError('Operator group inventory differs')
    actual=np.asarray([asnumpy(solver.ops[g].apply(solver.state[g])).reshape(-1) for g in range(groups)])
    actual=_finite(actual,(groups,n),'actual operator application')
    if data['operator_form']=='density':actual=actual*geo['volumes']
    scale=gross+abs(reaction)+abs(actual)
    error=abs(loss-actual)
    # Local error is scaled by the independently accumulated outgoing/reaction
    # terms, so cancellation to zero cannot hide a changed loss operator.
    effective_rtol=max(rtol,64*np.finfo(np.dtype(solver.dtype)).eps)
    if np.any(error>effective_rtol*scale):
        raise ValueError('Exported divergence/removal differs from actual diffusion operator')
    scaled=np.divide(error,scale,out=np.zeros_like(error),where=scale>0)
    return dict(interior=dict(owner=i.copy(),neighbor=j.copy(),owner_side=geo['sides'][:,0],
        neighbor_side=geo['sides'][:,1],area=areas,center=geo['centers'],normal=_NORMALS[geo['sides'][:,0]],
        normal_distances=dist.copy(),conductance=weights,integrated_current=integrated,
        neighbor_integrated_current=-integrated,current_density=integrated/areas,
        neighbor_current_density=-integrated/areas),
        boundary=dict(owner=b,side=geo['bsides'],area=bareas,center=geo['bcenters'],
            normal=_NORMALS[geo['bsides']],normal_distance=bd,alpha=alpha,
            conductance=bweights,integrated_current=boundary,current_density=boundary/bareas),
        cell_outgoing=outgoing,cell_reaction=reaction,integrated_loss=loss,operator_integrated_loss=actual,
        diagnostics=dict(operator_max_absolute_error_by_group=error.max(axis=1),
            operator_max_scaled_error_by_group=scaled.max(axis=1),requested_rtol=rtol,effective_rtol=effective_rtol,
            operator_form=data['operator_form'],all_six_sides_covered=True,
            interior_current_antisymmetry_exact=True,ADF='off',geometry='orthogonal_conforming_cuboids'))
