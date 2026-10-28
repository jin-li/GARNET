"""Independent spatial reference and physical error gates for TWIGL FTM."""

import numpy as np
import pytest
from scipy.linalg import expm

from garnet.stencil import GroupOperator
from tools.assess_exponential_twigl import (METHODS, Reference, TwiglSystem,
    error_gate, graded_widths, relative_blocks, restrict_field, run)


@pytest.fixture(scope='module')
def step_system():
    return TwiglSystem.build('step',1)


def test_independent_generator_matches_random_physical_kernel_actions(step_system):
    system = step_system
    count, grid = system.count, system.problem.grid
    materials, mmap = system.problem.problem_at(0.01)
    rng = np.random.default_rng(1949)
    state = rng.uniform(0.1,1.,3*count)
    flux, precursor = state[:2*count].reshape(2,*grid.shape), state[2*count:].reshape(grid.shape)
    source = np.sum(system.fission.reshape(2,*grid.shape)*flux,axis=0)
    expected = []
    for g in range(2):
        diffusion = np.array([m.diffusion[g] for m in materials])[mmap]
        removal = np.array([m.sigma_a[g]+m.sigma_s[g].sum() for m in materials])[mmap]
        loss = GroupOperator(np,grid,diffusion,removal,bc=system.problem.bc).apply(flux[g])
        rhs = -loss
        for other in range(2):
            if other != g:
                rhs = rhs + np.array([m.sigma_s[other,g] for m in materials])[mmap]*flux[other]
        spectrum = np.array([m.chi[g] for m in materials])[mmap]
        rhs = rhs + spectrum*((1-0.0075)*source+0.08*precursor)
        expected.append(system.problem.kinetics.velocities[g]*rhs)
    expected.append(0.0075*source-0.08*precursor)
    np.testing.assert_allclose(system.generator(0.,'step')@state,
                               np.asarray(expected).ravel(),rtol=3e-13,atol=1e-8)


def test_sparse_radau_step_reference_matches_dense_matrix_exponential(step_system):
    system = step_system
    reference = Reference.solve(system,'step',duration=0.002,rtol=3e-11)
    # An independent augmented exponential also integrates relative fission
    # power, checking the reference's dense-output quadrature separately.
    size = system.initial.size
    augmented = np.zeros((size+1,size+1))
    augmented[:size,:size] = system.generator(0.,'step').toarray()
    augmented[-1,:2*system.count] = system.fission.ravel()
    expected = expm(0.002*augmented)@np.r_[system.initial,0.]
    np.testing.assert_allclose(reference(0.002),expected[:-1],rtol=2e-8,atol=1e-12)
    assert reference.integral()==pytest.approx(expected[-1],rel=2e-8)
    assert float(system.power(reference(0.))) == pytest.approx(1.,abs=1e-14)


def test_ramp_reference_preserves_all_fields_across_its_control_corner():
    system = TwiglSystem.build('ramp',1)
    reference = Reference.solve(system,'ramp',duration=0.23,rtol=3e-10)
    assert len(reference.segments)==2
    np.testing.assert_array_equal(reference.segments[0].y[:,-1],reference.segments[1].y[:,0])
    np.testing.assert_allclose(reference(0.2),reference.segments[0].y[:,-1],rtol=5e-15)
    assert (system.generator(.2,'ramp')!=system.generator(.23,'ramp')).nnz==0
    assert reference.integral()>0.23


def test_area_restriction_conserves_mean_and_error_norm_separates_amplitude():
    field = np.arange(64.).reshape(8,8)
    coarse = restrict_field(field,4)
    assert coarse.mean()==field.mean()
    np.testing.assert_array_equal(restrict_field(np.ones((2,8,8)),4),np.ones((2,4,4)))
    expected = np.array([[1.,2.],[2.,4.]])
    actual = 2*expected
    np.testing.assert_allclose(relative_blocks(actual,expected),1.)
    np.testing.assert_allclose(relative_blocks(actual/2,expected),0.)
    with pytest.raises(ValueError,match='nested'):
        restrict_field(field,3)


@pytest.mark.parametrize('event',[None,.2])
def test_shared_fixed_mesh_is_positive_and_exactly_aligned(event):
    widths = graded_widths(event=event)
    assert np.all(widths>0)
    assert widths.sum()==pytest.approx(.5,abs=1e-15)
    if event is not None:
        assert np.min(abs(np.cumsum(widths)-event))<1e-14


@pytest.mark.parametrize('method',METHODS)
def test_actual_spatial_transients_meet_all_declared_physical_error_gates(step_system,method):
    reference = Reference.solve(step_system,'step',duration=.02,rtol=3e-10)
    result = run(step_system,reference,'step',method,1e-5)
    assert error_gate(result,.001),result
    assert result['implicit_solves']>=result['accepted_steps']
    # Power alone cannot qualify a run that misses spatial/precursor errors.
    contaminated = dict(result,final_relative_precursor_rms_error_by_family=[.01])
    assert not error_gate(contaminated,.001)
    contaminated = dict(result,maximum_relative_shape_rms_error_by_group=[.01,0])
    assert not error_gate(contaminated,.001)
