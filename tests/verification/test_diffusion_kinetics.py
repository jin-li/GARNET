"""Independent physical static FV kinetics checks, including off-criticality."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import eig

from ndgpu import DiffusionEigenSolver, Grid, Kinetics, Material, SP3EigenSolver
from ndgpu.diffusion_kinetics import project_diffusion_kinetics
import ndgpu.diffusion_kinetics as diffusion_kinetics
from ndgpu.quasistatic import project_effective_kinetics


def homogeneous():
    grid = Grid((3, 1, 1), (30., 1., 1.))
    mat = Material(name="one", diffusion=[1.3], sigma_a=[.03], nu_sigma_f=[.039])
    solver = DiffusionEigenSolver(grid, mat, device="cpu", bc="reflective")
    f = solver.solve(tol_k=1e-11, tol_source=1e-11)
    a = solver.solve(adjoint=True, tol_k=1e-11, tol_source=1e-11)
    kin = Kinetics([2.2e5], [.002, .004], [.08, .2])
    return solver, f, a, kin


def test_offcritical_one_group_and_normalization():
    solver, f, a, kin = homogeneous()
    p = project_diffusion_kinetics(solver, f, a, kin)
    assert p.k_eff == pytest.approx(1.3)
    assert p.generation_time == pytest.approx(1/(2.2e5*.039))
    assert p.lifetime == pytest.approx(1/(2.2e5*.03))
    assert p.lambda_low_frequency == pytest.approx(.006/(.002/.08+.004/.2))
    np.testing.assert_allclose(p.beta, kin.beta)
    assert p.beta_total == pytest.approx(.006)
    legacy = project_effective_kinetics(solver, f, a, kin)
    assert legacy.generation_time == pytest.approx(p.lifetime)
    q = project_diffusion_kinetics(solver, f.flux*3, a.flux*.017, kin,
                                  k_eff=f.k_eff, adjoint_k_eff=a.k_eff)
    assert q.generation_time == pytest.approx(p.generation_time)
    np.testing.assert_allclose(q.beta, p.beta)
    zero = project_diffusion_kinetics(solver, f, a,
                                     Kinetics([2.2e5], [0], [.1]))
    assert zero.lambda_low_frequency is None


def test_independent_cylindrical_multigroup_pencil():
    # Nonuniform radial volumes, two materials, upscatter and different spectra.
    grid = Grid((3, 1, 1), (6., 1., 1.), geometry="cylindrical")
    mats = [Material(name="a", diffusion=[1.2,.4], sigma_a=[.02,.08],
                     nu_sigma_f=[.015,.13], sigma_s=[[0,.025],[.004,0]], chi=[.9,.1]),
            Material(name="b", diffusion=[.8,.3], sigma_a=[.03,.065],
                     nu_sigma_f=[.022,.11], sigma_s=[[0,.04],[.007,0]], chi=[.8,.2])]
    ids = np.array([0,1,0])
    solver = DiffusionEigenSolver(grid, mats, ids[:,None,None],
                                 device="cpu", bc="reflective")
    f = solver.solve(tol_k=1e-11, tol_source=1e-11)
    a = solver.solve(adjoint=True, tol_k=1e-11, tol_source=1e-11)
    kin = Kinetics([[2e7,2e5],[1.8e7,2.4e5]], [[.002,.004],[.001,.006]],
                   [.08,.2], [[.65,.35],[.4,.6]])
    A, F, T = np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6))
    delayed = np.zeros((2,6,6))
    radius = np.array([1.,3.,5.])
    for n, m in enumerate(ids):
        mat = mats[m]
        for g in range(2):
            i = g*3+n
            A[i,i] = radius[n]*(mat.sigma_a[g] + mat.sigma_s[g].sum())
            T[i,i] = radius[n]/kin.velocities[m,g]
            for h in range(2):
                j = h*3+n
                if h != g:
                    A[i,j] = -radius[n]*mat.sigma_s[h,g]
                F[i,j] = radius[n]*mat.chi[g]*mat.nu_sigma_f[h]
                for family in range(2):
                    delayed[family,i,j] = (radius[n]*kin.chi_delayed[family,g]
                                          *kin.beta[m,family]*mat.nu_sigma_f[h])
    for g in range(2):
        for n in range(2):
            d0,d1 = mats[ids[n]].diffusion[g],mats[ids[n+1]].diffusion[g]
            coupling = (2*d0*d1/(d0+d1))*(2*(n+1))/4
            i,j = g*3+n,g*3+n+1
            A[i,i] += coupling
            A[j,j] += coupling
            A[i,j] -= coupling
            A[j,i] -= coupling
    values, left, right = eig(A,F,left=True,right=True)
    ix = min((i for i,v in enumerate(values) if np.isfinite(v) and
              abs(v.imag)<1e-10 and v.real>0), key=lambda i:values[i].real)
    k = 1/values[ix].real
    phi,star = right[:,ix].real,left[:,ix].real
    assert f.converged and a.converged
    assert f.k_eff == pytest.approx(k,rel=1e-9)
    np.testing.assert_allclose(f.flux.ravel()/f.flux.ravel()[0], phi/phi[0],rtol=1e-8)
    np.testing.assert_allclose(a.flux.ravel()/a.flux.ravel()[0],star/star[0],rtol=1e-8)
    p = project_diffusion_kinetics(solver,f,a,kin)
    denominator = star@F@phi
    assert p.generation_time == pytest.approx(star@T@phi/denominator,rel=1e-8)
    np.testing.assert_allclose(p.beta,[star@d@phi/denominator for d in delayed],rtol=1e-8)
    assert p.forward_residual < 1e-8 and p.adjoint_residual < 1e-8


def test_reject_unqualified_pairs_and_models():
    solver,f,a,kin = homogeneous()
    with pytest.raises(ValueError,match="not converged"):
        project_diffusion_kinetics(solver,replace(f,converged=False),a,kin)
    with pytest.raises(ValueError,match="disagree"):
        project_diffusion_kinetics(solver,f,replace(a,k_eff=a.k_eff*1.01),kin)
    bad = a.flux.copy()
    bad[0,0,0,0] *= 1.1
    with pytest.raises(ValueError,match="complete diffusion"):
        project_diffusion_kinetics(solver,f,bad,kin,adjoint_k_eff=a.k_eff)
    with pytest.raises(ValueError,match="complete diffusion"):
        project_diffusion_kinetics(solver,bad,a,kin,k_eff=f.k_eff)
    with pytest.raises(ValueError,match="eigenvalue must be supplied"):
        project_diffusion_kinetics(solver,f.flux,a,kin)
    sp3 = SP3EigenSolver(solver.grid, Material(name="a", diffusion=[1.],
                           sigma_a=[.03],nu_sigma_f=[.04]),device="cpu")
    with pytest.raises(TypeError,match="standard symmetric FV"):
        project_diffusion_kinetics(sp3,f,a,kin)


@pytest.mark.parametrize("field,value", [("velocities",np.nan), ("beta",np.inf),
                                        ("decay",0.)])
def test_invalid_kinetics_are_rejected(field,value):
    solver,f,a,kin = homogeneous()
    getattr(kin,field)[0] = value
    with pytest.raises(ValueError,match="finite|positive"):
        project_diffusion_kinetics(solver,f,a,kin)


@pytest.mark.parametrize("forward_scale,adjoint_scale", [(1e200,1e200),
                         (1e-200,1e-200),(1e200,1e-200),(1e-200,1e200)])
def test_extreme_normalizations_preserve_physical_projection(forward_scale,adjoint_scale):
    solver,f,a,kin = homogeneous()
    p = project_diffusion_kinetics(solver,f,a,kin)
    q = project_diffusion_kinetics(solver,f.flux*forward_scale,a.flux*adjoint_scale,
                                  kin,k_eff=f.k_eff,adjoint_k_eff=a.k_eff)
    assert q.generation_time == pytest.approx(p.generation_time)
    assert q.lifetime == pytest.approx(p.lifetime)
    np.testing.assert_allclose(q.beta,p.beta)


@pytest.mark.parametrize("scale", [1e200,1e-200])
def test_invalid_extreme_shapes_cannot_bypass_residual_checks(scale):
    solver,f,a,kin = homogeneous()
    bad = f.flux.copy()
    bad[0,0,0,0] *= 1.1
    with pytest.raises(ValueError,match="complete diffusion"):
        project_diffusion_kinetics(solver,bad*scale,a.flux*scale,kin,
                                   k_eff=f.k_eff,adjoint_k_eff=a.k_eff)


def test_negative_prompt_remainder_is_rejected():
    grid = Grid((2,1,1),(20.,1.,1.))
    mat = Material(name="total fast spectrum", diffusion=[1.,.4],
                   sigma_a=[.03,.05],nu_sigma_f=[.01,.1],
                   sigma_s=[[0,.02],[.003,0]],chi=[1.,0.])
    solver = DiffusionEigenSolver(grid,mat,device="cpu",bc="reflective")
    f = solver.solve(tol_k=1e-11,tol_source=1e-11)
    a = solver.solve(adjoint=True,tol_k=1e-11,tol_source=1e-11)
    for beta in ([.006], [[.006]]):
        kin = Kinetics([2e7,2e5],beta,[.08],[.5,.5])
        with pytest.raises(ValueError,match="prompt remainder"):
            project_diffusion_kinetics(solver,f,a,kin)


def test_nonfinite_operator_diagnostics_rejected(monkeypatch):
    solver,f,a,kin = homogeneous()
    monkeypatch.setattr(solver.ops[0],"apply",lambda flux: np.full_like(flux,np.nan))
    with pytest.raises(ValueError,match="nonfinite complete diffusion"):
        project_diffusion_kinetics(solver,f,a,kin)


@pytest.mark.parametrize("field", ["generation_time","time_importance",
                                 "fission_importance","beta"])
def test_nonfinite_projection_outputs_rejected(monkeypatch,field):
    solver,f,a,kin = homogeneous()
    native = diffusion_kinetics.project_effective_kinetics
    def invalid(*args,**kwargs):
        result = SimpleNamespace(**vars(native(*args,**kwargs)))
        setattr(result,field,np.array([np.nan]) if field == "beta" else np.nan)
        return result
    monkeypatch.setattr(diffusion_kinetics,"project_effective_kinetics",invalid)
    with pytest.raises(ValueError,match="nonfinite or unphysical"):
        project_diffusion_kinetics(solver,f,a,kin)
