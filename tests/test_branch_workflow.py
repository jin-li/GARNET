"""End-to-end independent homogeneous limit for the portable branch example."""
from pathlib import Path
import runpy
import numpy as np
import pytest


def test_triangle_evaluation_through_fresh_nem_has_analytic_eigenpair():
    example=Path(__file__).resolve().parents[1]/'examples/isonees_features/branch_xs.py'
    library,state,solver,result,exact,report=runpy.run_path(str(example))['solve']()
    assert state.interpolated and state.qualification == 'interpolated-unvalidated'
    assert len(state.contributing_ids) == 3
    assert all(w >= 0 for w in state.weights)
    assert sum(state.weights) == pytest.approx(1)
    assert result.k_eff == pytest.approx(exact,rel=1e-10)
    assert report['status'] == 'pass'
    np.testing.assert_allclose(result.means/result.means[0],1,rtol=1e-10)
    np.testing.assert_allclose(result.currents,0,atol=1e-12)
    # A rectangle corner is outside the star even though each coordinate is
    # individually bracketed. Rejecting it prevents accidental extrapolation.
    with pytest.raises(ValueError,match='outside'):
        library.evaluate(dict(configuration='synthetic',family='thermal',
                         fuel_temperature=700.,moderator_density=.69325))


def test_thermal_star_example_recovers_independent_homogeneous_limit():
    example = Path(__file__).resolve().parents[1] / 'examples/branch_xs/thermal_star.py'
    library, state, solver, result, exact, report = runpy.run_path(str(example))['solve']()
    assert len(library.simplices) == 4
    assert len(library.states) == 5
    assert state.qualification == 'interpolated-unvalidated'
    assert all(s.qualification == 'diagnostic-unvalidated' for s in library.states.values())
    assert result.k_eff == pytest.approx(exact, rel=1e-10)
    np.testing.assert_allclose(result.means/result.means[0], 1., rtol=1e-10)
    np.testing.assert_allclose(result.currents, 0., atol=1e-12)
    assert report['status'] == 'pass'
