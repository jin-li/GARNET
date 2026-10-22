"""3D radial/axial transmission MMS on conforming triangular prisms.

The circular reference is analytically averaged using each material's radial
quadratic extension. Inscribed interface and boundary chords introduce O(h^2)
geometric defects; this verifies joint geometry/discretization convergence.
"""
import resource
import time

import numpy as np
from scipy.sparse.linalg import spsolve

from ndgpu.corrected_mesh import CorrectedMeshGroupOperator
from ndgpu.mesh import MeshGrid, assemble_mesh, assemble_mesh_3d
from test_corrected_mesh_radial_mms import RADII, radial_cell_averages


def extruded_polar_mesh(refinement, half_height):
    angles = 24 * refinement
    counts = np.array([2, 1, 1]) * refinement
    theta = 2*np.pi*np.arange(angles)/angles
    unit = np.column_stack((np.cos(theta), np.sin(theta)))
    radii, materials = [], []
    for material, (inner, outer, count) in enumerate(
            zip(np.r_[0., RADII[:-1]], RADII, counts)):
        radii.extend(np.linspace(inner, outer, count+1)[1:])
        materials.extend([material]*count)
    xy = np.vstack((np.zeros((1, 2)), *(r*unit for r in radii)))
    triangles, tags = [], []
    for ring, material in enumerate(materials):
        for j in range(angles):
            a = 1 + ring*angles + j
            b = 1 + ring*angles + (j+1) % angles
            if ring == 0:
                triangles.append((0, a, b))
                tags.append(material)
            else:
                triangles.extend(((a-angles, a, b), (a-angles, b, b-angles)))
                tags.extend((material, material))
    planar = assemble_mesh(xy, triangles, np.asarray(tags))
    slabs = 2*refinement
    z = np.linspace(-half_height, half_height, slabs+1)
    points = np.vstack([np.column_stack((xy, np.full(len(xy), height))) for height in z])
    prisms = [tuple(node+k*len(xy) for node in tri)
              + tuple(node+(k+1)*len(xy) for node in tri)
              for k in range(slabs) for tri in triangles]
    mesh = assemble_mesh_3d(points, prisms, np.tile(tags, slabs), normal_distances=True)
    return mesh, planar, xy, z, angles


def test_extruded_radial_axial_transmission_mms():
    D = np.array([.7, 1800., .3])
    H = RADII[-1]
    errors, outflow_errors = [], []
    for refinement in (1, 2, 4):
        start = time.perf_counter()
        mesh, planar, xy, z, angles = extruded_polar_mesh(refinement, H)
        slabs = len(z)-1
        expected_volumes = np.tile(planar.area, slabs)*np.repeat(np.diff(z), planar.n_cells)
        np.testing.assert_allclose(mesh.area, expected_volumes, rtol=2e-12)
        polygon_area = angles*RADII[-1]**2*np.sin(2*np.pi/angles)/2
        np.testing.assert_allclose(mesh.area.sum(), 2*H*polygon_area, rtol=2e-13)
        for normals in (mesh.face_normals, mesh.bface_normals):
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1., atol=2e-15)
        for (i, j, _, _), center, normal in zip(
                mesh.faces, mesh.face_centers, mesh.face_normals):
            assert np.dot(center-mesh.centroid[i], normal) > 0
            assert np.dot(mesh.centroid[j]-center, normal) > 0
        for (i, _, _), center, normal in zip(
                mesh.bfaces, mesh.bface_centers, mesh.bface_normals):
            assert np.dot(center-mesh.centroid[i], normal) > 0
        radial_mean = radial_cell_averages(planar, xy, D)
        axial_mean = 1-(z[:-1]**2 + z[:-1]*z[1:] + z[1:]**2)/(3*H**2)
        exact = (axial_mean[:, None]*radial_mean).ravel()
        # -div(D grad u) = (1-z^2/H^2) + 2D*u_rad/H^2.
        # Both terms have independent analytic prism integrals.
        source = (axial_mean[:, None]
                  + 2*D[planar.cell_tag]*radial_mean/H**2).ravel()
        rhs = mesh.area*source
        grid = MeshGrid(mesh, "zero-flux")
        op = CorrectedMeshGroupOperator(np, grid, D[mesh.cell_tag], np.zeros(mesh.n_cells))
        solution = spsolve(op.host_matrix.tocsc(), rhs)
        error = np.sqrt(np.sum(mesh.area*(solution-exact)**2))
        errors.append(error)
        np.testing.assert_allclose(op.apply(solution).ravel(), rhs, rtol=2e-8, atol=2e-10)
        interior, boundary = op.face_currents(solution)
        balance = np.zeros(mesh.n_cells)
        for (i, j, _, _), current in zip(mesh.faces, interior):
            balance[i] += current
            balance[j] -= current
        for (i, _, _), current in zip(mesh.bfaces, boundary):
            balance[i] += current
        np.testing.assert_allclose(balance, rhs, rtol=2e-8, atol=2e-10)
        np.testing.assert_allclose(boundary.sum(), rhs.sum(), rtol=2e-9)
        # Analytic flux: J_xy=x*g(z)/2; J_z=2D*u_rad*z/H^2.
        # Its integrated polygon outflow equals the analytic volume source.
        exact_side = polygon_area*4*H/3
        exact_caps = 4/H*np.sum(planar.area*D[planar.cell_tag]*radial_mean)
        np.testing.assert_allclose(rhs.sum(), exact_side+exact_caps, rtol=2e-13)
        caps = np.abs(mesh.bface_normals[:, 2]) > .5
        outflow_errors.append(np.abs([boundary[~caps].sum()-exact_side,
                                     boundary[caps].sum()-exact_caps]))
        sagitta = RADII[-1]*(1-np.cos(np.pi/angles))
        # Max side trace defect is at z=0 and a boundary chord midpoint.
        boundary_defect = RADII[-1]**2*np.sin(np.pi/angles)**2/(4*D[-1])
        print(f"prisms={mesh.n_cells} weighted_L2={error:.9e} "
              f"outflow={boundary.sum():.12e} "
              f"side_outflow={boundary[~caps].sum():.9e} exact_side={exact_side:.9e} "
              f"cap_outflow={boundary[caps].sum():.9e} exact_caps={exact_caps:.9e} "
              f"sagitta={sagitta:.9e} boundary_trace_defect={boundary_defect:.9e} "
              f"seconds={time.perf_counter()-start:.3f} "
              f"peak_RSS_MiB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f}")
    ratios = np.asarray(errors[:-1])/errors[1:]
    outflow_ratios = np.asarray(outflow_errors[:-1])/outflow_errors[1:]
    print(f"error_ratios={ratios.tolist()} orders={np.log2(ratios).tolist()} "
          f"side_cap_outflow_orders={np.log2(outflow_ratios).tolist()}")
    assert np.all(ratios > 3.), f"3D radial/axial MMS failed: {errors}, {ratios}"
    assert np.all(outflow_ratios > 3.), f"3D outflow refinement failed: {outflow_errors}"
