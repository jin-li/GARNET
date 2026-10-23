"""Circular-interface MMS on independently generated, inscribed polar meshes.

The reference is the circular-domain solution, integrated analytically over
each triangle using its material's quadratic extension. Chord interfaces and
the outer polygon therefore introduce O(dtheta**2) geometric consistency
errors; this is a joint geometry/discretization refinement, not a claim that
the circular solution exactly satisfies the polygonal problem.
"""
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from garnet.corrected_mesh import CorrectedMeshGroupOperator
from garnet.mesh import MeshGrid, assemble_mesh


RADII = np.array([.4107, .4186, .6309])


def polar_mesh(refinement):
    """Conforming triangles, with a separate ring at every material radius."""
    angles = 48 * refinement
    counts = np.array([4, 1, 3]) * refinement
    rings, ring_tags = [], []
    for material, (inner, outer, count) in enumerate(
            zip(np.r_[0., RADII[:-1]], RADII, counts)):
        rings.extend(np.linspace(inner, outer, count + 1)[1:])
        ring_tags.extend([material] * count)
    theta = 2 * np.pi * np.arange(angles) / angles
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    coords = np.vstack((np.zeros((1, 2)),
                        *(radius * directions for radius in rings)))
    cells, tags = [], []
    for ring, material in enumerate(ring_tags):
        for j in range(angles):
            outer = 1 + ring * angles + j
            outer_next = 1 + ring * angles + (j + 1) % angles
            if ring == 0:
                cells.append((0, outer, outer_next))
                tags.append(material)
            else:
                inner = outer - angles
                inner_next = outer_next - angles
                cells.extend(((inner, outer, outer_next),
                              (inner, outer_next, inner_next)))
                tags.extend((material, material))
    return assemble_mesh(coords, cells, np.asarray(tags)), coords, angles


def radial_cell_averages(mesh, coords, diffusion):
    # u_m(r) = C_m - r^2/(4 D_m), continuous at the circular interfaces.
    inner = np.r_[0., RADII[:-1]]
    drops = (RADII**2 - inner**2) / (4 * diffusion)
    constants = np.cumsum(drops[::-1])[::-1] + inner**2 / (4 * diffusion)
    vertices = coords[np.asarray(mesh.cells)]
    # Triangle average of x.x = (sum(v_i.v_i) + |sum(v_i)|^2)/12.
    mean_r2 = (np.sum(vertices**2, axis=(1, 2))
               + np.sum(vertices.sum(axis=1)**2, axis=1)) / 12
    return constants[mesh.cell_tag] - mean_r2 / (4 * diffusion[mesh.cell_tag])


@pytest.mark.parametrize("diffusion", [(0.7, 3., 0.3), (0.7, 1800., 0.3)],
                         ids=["moderate", "thin-gap-high-contrast"])
def test_curved_radial_mms(diffusion):
    diffusion = np.asarray(diffusion)
    errors = []
    for refinement in (1, 2, 4):
        mesh, coords, angles = polar_mesh(refinement)
        grid = MeshGrid(mesh, "zero-flux")
        op = CorrectedMeshGroupOperator(np, grid, diffusion[mesh.cell_tag],
                                        np.zeros(mesh.n_cells))
        rhs = mesh.area.copy()  # -div(D grad u) = 1, integrated exactly.
        polygon_area = angles * RADII[-1]**2 * np.sin(2 * np.pi / angles) / 2
        np.testing.assert_allclose(mesh.area.sum(), polygon_area, rtol=2e-13)
        solution = spsolve(op.host_matrix.tocsc(), rhs)
        exact = radial_cell_averages(mesh, coords, diffusion)
        error = np.sqrt(np.sum(mesh.area * (solution - exact)**2))
        errors.append(error)
        np.testing.assert_allclose(op.apply(solution).ravel(), rhs,
                                   rtol=2e-8, atol=2e-11)
        interior, boundary = op.face_currents(solution)
        # Every interior current cancels with its equal-and-opposite neighbor.
        balance = np.zeros(mesh.n_cells)
        for (i, j, _, _), current in zip(mesh.faces, interior):
            balance[i] += current
            balance[j] -= current
        for (i, _, _), current in zip(mesh.bfaces, boundary):
            balance[i] += current
        np.testing.assert_allclose(balance, rhs, rtol=2e-8, atol=2e-11)
        np.testing.assert_allclose(boundary.sum(), mesh.area.sum(), rtol=2e-9)
        # The analytic current is x/2, so its polygon outflow equals area.
        analytic_outflow = sum(area * np.dot(center, normal) / 2
                               for (_, area, _), center, normal in zip(
                                   mesh.bfaces, mesh.bface_centers,
                                   mesh.bface_normals))
        np.testing.assert_allclose(analytic_outflow, mesh.area.sum(), rtol=2e-13)
        sagitta = RADII[-1] * (1 - np.cos(np.pi / angles))
        # Maximum outer boundary mismatch of the circular exact solution.
        boundary_defect = RADII[-1]**2 * np.sin(np.pi / angles)**2 / (4 * diffusion[-1])
        # At each chord midpoint, the two material-wise quadratic extensions
        # differ by this amount; their normal currents still agree exactly.
        interface_defects = (RADII[:-1]**2 * np.sin(np.pi / angles)**2 / 4
                             * np.abs(1 / diffusion[:-1] - 1 / diffusion[1:]))
        print(f"D={diffusion.tolist()} cells={mesh.n_cells} angles={angles} "
              f"weighted_L2={error:.9e} sagitta={sagitta:.9e} "
              f"boundary_trace_defect={boundary_defect:.9e} "
              f"interface_trace_defects={interface_defects.tolist()} "
              f"outflow={boundary.sum():.12e}")
    ratios = np.asarray(errors[:-1]) / errors[1:]
    print(f"error_ratios={ratios.tolist()} orders={np.log2(ratios).tolist()}")
    assert np.all(ratios > 3.), f"radial curved-interface MMS failed: {errors}, {ratios}"
