"""Synthetic exact reflection folding; no reactor XS or assembly solves."""
import numpy as np
import pytest

from garnet.corrected_native_moments import CorrectedNativeMomentOperator
from garnet.mesh import MeshGrid, assemble_mesh
from test_native_sp5 import material


def mirrored_meshes(angle=0.):
    # A perturbed interior vertex makes tangential reconstruction observable;
    # orthogonal rectangles would hide gradient errors in the action.
    coords = np.array([(0.,0.),(.5,0.),(1.,0.),(0.,.5),(.43,.57),(1.,.5),
                       (0.,1.),(.5,1.),(1.,1.)])
    cells = [(0,1,4,3),(1,2,5,4),(3,4,7,6),(4,5,8,7)]
    full_coords = coords.tolist(); mirror = {}
    for i,(x,y) in enumerate(coords):
        if x == 0:
            mirror[i] = i
        else:
            mirror[i] = len(full_coords); full_coords.append([-x,y])
    full_cells = cells + [tuple(mirror[k] for k in reversed(cell)) for cell in cells]
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    half = assemble_mesh(coords@rotation.T,cells,np.arange(4))
    full = assemble_mesh(np.asarray(full_coords)@rotation.T,full_cells,np.tile(np.arange(4),2))
    reflection = rotation@np.diag([-1.,1.])@rotation.T
    # These geometric relationships define the embedding independently of the
    # native operator's reconstruction/face code.
    np.testing.assert_allclose(full.centroid[:4],half.centroid,atol=1e-14)
    np.testing.assert_allclose(full.centroid[4:],half.centroid@reflection.T,atol=1e-14)
    np.testing.assert_allclose(full.area,np.tile(half.area,2),atol=1e-14)
    for mesh in (half,full):
        assert np.min(mesh.face_distances)>0
        for (i,j,*_),center,normal,distances in zip(mesh.faces,mesh.face_centers,mesh.face_normals,mesh.face_distances):
            np.testing.assert_allclose(distances,[(center-mesh.centroid[i])@normal,
                (mesh.centroid[j]-center)@normal],atol=1e-14)
    return half,full,rotation,reflection


def operators(family,angle,exterior):
    half,full,rotation,reflection = mirrored_meshes(angle)
    normal = rotation@np.array([-1.,0.])
    cut = [abs(center@normal)<1e-13 for center in half.bface_centers]
    assert sum(cut)==2
    half_bc = ['reflective' if flag else exterior for flag in cut]
    # Four distinct material matrices are copied into their geometric mirrors.
    # Adjacent full matrices do not commute; the mirror neighbor itself shares D.
    materials = [material(i,groups=2) for i in range(4)]
    a = CorrectedNativeMomentOperator(MeshGrid(half,half_bc),materials,family=family)
    b = CorrectedNativeMomentOperator(MeshGrid(full,exterior),materials+materials,family=family)
    for diffusion in a.diffusion:
        assert np.linalg.norm(diffusion-np.diag(np.diag(diffusion)))>1e-4
    assert np.linalg.norm(a.diffusion[0]@a.diffusion[1]-a.diffusion[1]@a.diffusion[0])>1e-5
    return a,b,rotation,reflection,cut


def selected_currents(a,b,interior,boundary,cut):
    rows = {tuple(sorted((i,j))):(k,i,j) for k,(i,j,*_) in enumerate(b.grid.mesh.faces)}
    common = []
    for i,j,*_ in a.grid.mesh.faces:
        k,left,right = rows[tuple(sorted((i,j)))]
        common.append(interior[k] if (i,j)==(left,right) else -interior[k])
    exterior = []; across_cut = []
    for k,((i,*_),center) in enumerate(zip(a.grid.mesh.bfaces,a.grid.mesh.bface_centers)):
        if cut[k]:
            index,left,right = rows[(i,4+i)]
            value = interior[index] if left==i else -interior[index]
            across_cut.append(value); exterior.append(value)
        else:
            matches = [j for j,((owner,*_),other) in enumerate(zip(b.grid.mesh.bfaces,b.grid.mesh.bface_centers))
                       if owner==i and np.allclose(center,other,rtol=0,atol=1e-13)]
            assert len(matches)==1
            exterior.append(boundary[matches[0]])
    return np.asarray(common),np.asarray(exterior),np.asarray(across_cut)


@pytest.mark.parametrize('family',['sp3','sdp1','sp5'])
@pytest.mark.parametrize('angle',[0.,-45.])
@pytest.mark.parametrize('exterior',['reflective','vacuum'])
def test_exact_mirrored_heterogeneous_full_matrix_and_every_basis_map(family,angle,exterior):
    a,b,rotation,reflection,cut = operators(family,angle,exterior)
    # Native components are scalar under spatial reflection. E copies cell
    # values; S selects identical physical cells. Their areas are unchanged.
    identity = np.eye(a.size)
    E = np.vstack((identity,identity))
    S = np.hstack((identity,np.zeros_like(identity)))
    np.testing.assert_allclose(S@(b.assembled@E),a.assembled.toarray(),rtol=3e-12,atol=3e-13)
    gradient_half=[];gradient_full=[];current_half=[];current_full=[]
    for vector in identity:
        extended = E@vector
        ah,bh = a.gradients(vector),b.gradients(extended)
        np.testing.assert_allclose(bh[4:],bh[:4]@reflection,rtol=3e-12,atol=3e-13)
        gradient_half.append(ah);gradient_full.append(bh[:4])
        ai,ab = a.face_currents(vector);bi,bb = b.face_currents(extended)
        selected_i,selected_b,across_cut = selected_currents(a,b,bi,bb,cut)
        current_half.append(np.r_[ai.ravel(),ab.ravel()])
        current_full.append(np.r_[selected_i.ravel(),selected_b.ravel()])
        np.testing.assert_allclose(across_cut,0,rtol=0,atol=3e-13)
        np.testing.assert_array_equal(ab[np.asarray(cut)],0)
        np.testing.assert_allclose(a.action(vector),S@b.action(extended),rtol=3e-12,atol=3e-13)
        np.testing.assert_allclose(a.assembled@vector,a.action(vector),rtol=3e-12,atol=3e-13)
        np.testing.assert_allclose(b.assembled@extended,b.action(extended),rtol=3e-12,atol=3e-13)
    np.testing.assert_allclose(gradient_half,gradient_full,rtol=3e-12,atol=3e-13)
    np.testing.assert_allclose(current_half,current_full,rtol=3e-12,atol=3e-13)
    # No strong H*n=0 assertion: the Neumann row remains part of a soft LS fit.


@pytest.mark.parametrize('family',['sp3','sdp1','sp5'])
def test_oblique_mixed_boundary_rotation_covariance(family):
    a,_,_,_,_ = operators(family,0.,'vacuum')
    rotated,_,rotation,_,_ = operators(family,-45.,'vacuum')
    np.testing.assert_allclose(rotated.assembled.toarray(),a.assembled.toarray(),rtol=3e-12,atol=3e-13)
    for vector in np.eye(a.size):
        np.testing.assert_allclose(rotated.gradients(vector),a.gradients(vector)@rotation.T,rtol=3e-12,atol=3e-13)
        for observed,expected in zip(rotated.face_currents(vector),a.face_currents(vector)):
            np.testing.assert_allclose(observed,expected,rtol=3e-12,atol=3e-13)
