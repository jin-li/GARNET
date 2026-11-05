"""Reference scalar-flux algebra, without transport or uncertainty qualification."""
import numpy as np
import pytest

from garnet.moment_energy_homogenization import collapse_native_energy_coefficients


def fields(groups=3, orders=4):
    result = dict(INF_TOT=np.arange(groups, dtype=float)+2,
                  INF_NSF=np.arange(groups, dtype=float)*.1,
                  INF_FISS=np.arange(groups, dtype=float)*.05+.1,
                  INF_KAPPA=np.arange(groups, dtype=float)*10+190,
                  INF_RABSXS=-np.arange(groups, dtype=float)-.2,
                  INF_CHIT=np.arange(groups, dtype=float)*.2+.3,
                  INF_ABS=np.arange(groups, dtype=float)*.3)
    for order in range(orders):
        a = (np.arange(groups*groups, dtype=float).reshape(groups, groups)+1)/20
        if order:
            a[0, -1] *= -2
            a[-1, 0] *= -3
        result[f'INF_S{order}'] = a/(order+1)
        result[f'INF_SP{order}'] = a.T*.7/(order+1)
    return result


def collapse(data=None, flux=None, grid=None, partition=((0,), (1, 2)), **kwargs):
    return collapse_native_energy_coefficients(
        fields() if data is None else data, [2., 3., 5.] if flux is None else flux,
        [20., 1., .1, 0.] if grid is None else grid, partition, **kwargs)


def test_asymmetric_from_major_reference_rates_losses_source_and_power():
    data = fields()
    flux = np.array([2., 3., 5.])
    blocks = ((0,), (1, 2))
    result = collapse(data, flux)
    support = np.array([2., 8.])
    np.testing.assert_allclose(result.total, [2., (3*3+5*4)/8])
    for key in ('INF_TOT', 'INF_NSF', 'INF_FISS', 'INF_RABSXS', 'INF_ABS'):
        expected = [sum(flux[g]*data[key][g] for g in block) for block in blocks]
        np.testing.assert_allclose(result.raw[key]*support, expected)
    for order in range(4):
        for kind, matrices in (('S', result.collision), ('SP', result.production)):
            native = data[f'INF_{kind}{order}']
            expected = np.array([[sum(flux[g]*native[g, h] for g in incoming for h in outgoing)
                                  for outgoing in blocks] for incoming in blocks])
            np.testing.assert_allclose(matrices[order]*support[:, None], expected)
            assert not np.allclose(matrices[order], matrices[order].T)
        original_loss = np.diag(data['INF_TOT'])-data[f'INF_SP{order}'].T
        original_loss_rate = original_loss@flux
        expected_loss_rate = [sum(original_loss_rate[g] for g in block) for block in blocks]
        np.testing.assert_allclose(result.loss[order]@support, expected_loss_rate)
        np.testing.assert_allclose(result.loss[order], np.diag(result.total)-result.production[order].T)
    assert result.collision[2][0, 1] < 0
    assert result.production[3][0, 1] < 0
    assert np.all(result.raw['INF_RABSXS'] < 0)
    np.testing.assert_allclose(result.chi, [.3, 1.2])
    original_source = data['INF_NSF']@flux
    np.testing.assert_allclose(result.chi*(result.nu_sigma_f@support),
                               [.3*original_source, 1.2*original_source])
    assert result.chi.sum() == pytest.approx(1.5)
    expected_power = [sum(flux[g]*data['INF_FISS'][g]*data['INF_KAPPA'][g] for g in block)
                      for block in blocks]
    np.testing.assert_allclose(result.kappa_fission*support, expected_power)
    np.testing.assert_allclose(result.raw['INF_FISS']*result.raw['INF_KAPPA'], result.kappa_fission)
    wrong_kappa = sum(flux[1:]*data['INF_KAPPA'][1:])/support[1]
    assert result.kappa_fission[1] != pytest.approx(result.raw['INF_FISS'][1]*wrong_kappa)
    original_balance = data['INF_TOT']-data['INF_RABSXS']-data['INF_SP0'].sum(axis=1)
    expected_balance = [sum(flux[g]*original_balance[g] for g in block)/weight
                        for block, weight in zip(blocks, support)]
    np.testing.assert_allclose(result.balance_diagnostics['total_minus_reduced_absorption_production_out'],
                               expected_balance)


def test_23_to_22_bottom_pair_supports_zero_flux_zero_total_without_donor_filling():
    data = fields(23)
    flux = np.arange(23, dtype=float)+1
    flux[-1] = 0
    data['INF_TOT'][-1] = 0
    partition = tuple((g,) for g in range(21))+((21, 22),)
    result = collapse(data, flux, np.linspace(23., 0., 24), partition)
    assert result.n_groups == 22
    assert result.total[-1] == data['INF_TOT'][-2]
    assert result.provenance['original_zero_support_indices'] == (22,)
    assert result.chi[-1] == data['INF_CHIT'][-2:].sum()
    np.testing.assert_array_equal(result.provenance['energy_bounds_mev'], list(range(23, 1, -1))+[0])
    assert not result.provenance['donor_filling']
    assert flux[-1] == 0 and data['INF_TOT'][-1] == 0
    with pytest.raises(ValueError, match='aggregate flux support'):
        collapse(data, flux, np.linspace(23., 0., 24), tuple((g,) for g in range(23)))


def test_nonfissile_zero_chi_and_undefined_effective_kappa():
    data = fields()
    data['INF_FISS'][:] = 0
    data['INF_NSF'][:] = 0
    data['INF_CHIT'][:] = 0
    result = collapse(data)
    np.testing.assert_array_equal(result.chi, [0., 0.])
    np.testing.assert_array_equal(result.kappa_fission, [0., 0.])
    assert 'INF_KAPPA' not in result.raw
    assert not result.provenance['raw_kappa_defined']
    data['INF_FISS'][0] = .2
    result = collapse(data)
    assert result.kappa_fission[0] > 0
    assert 'INF_KAPPA' not in result.raw


def test_immutable_means_and_provenance_with_no_fabricated_errors():
    data = fields()
    source = {'source': 'native-results', 'nested': {'errors': np.array([.02])}}
    result = collapse(data, source_provenance=source)
    assert result.reported_errors == {}
    p = result.provenance
    assert p['aggregate_uncertainty'] == 'unknown'
    assert p['derived_coefficient_means']
    for flag in ('raw_fields_are_native_estimates', 'covariance_inferred', 'donor_errors_copied',
                 'uncertainty_qualification', 'physical_admission', 'production_admission',
                 'higher_estimator_transport_consistency_validated', 'angular_moment_rate_preservation'):
        assert not p[flag]
    for array in (result.total, result.collision[1], result.loss[3], result.raw['INF_FISS'],
                  p['original_energy_bounds_mev'], p['volume_integrated_scalar_flux'],
                  p['source_provenance']['nested']['errors']):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        p['source_provenance']['nested']['errors'] = []
    data['INF_TOT'][:] = 100
    source['nested']['errors'][:] = .9
    assert result.total[0] == 2
    assert p['source_provenance']['nested']['errors'][0] == .02


@pytest.mark.parametrize('field', ['INF_TOT', 'INF_NSF', 'INF_FISS', 'INF_KAPPA',
                                  'INF_CHIT', 'INF_ABS', 'INF_S0', 'INF_SP0'])
def test_unsigned_means_refuse_negatives(field):
    data = fields()
    data[field].flat[-1] = -1
    with pytest.raises(ValueError, match='Nonnegative'):
        collapse(data)


@pytest.mark.parametrize('field', ['INF_TOT', 'INF_NSF', 'INF_FISS', 'INF_KAPPA',
                                  'INF_RABSXS', 'INF_CHIT', 'INF_S0', 'INF_SP0'])
def test_required_fields_missing(field):
    data = fields()
    del data[field]
    with pytest.raises(ValueError):
        collapse(data)


@pytest.mark.parametrize('partition', [(), ((0,), (), (1, 2)), ((0,), (2, 1)), ((0,), (1,)),
                                        ((0,), (1, 1, 2)), ((0,), (1, 2, 3)),
                                        ((False,), (1, 2)), ((0.,), (1, 2)), (0, 1, 2), None])
def test_invalid_partition(partition):
    with pytest.raises(ValueError, match='partition'):
        collapse(partition=partition)


@pytest.mark.parametrize('grid', [[20., 1., 1., 0.], [20., 1., 2., 0.],
                                  [20., 1., .1, -1.], [20., 1., 0.],
                                  [20., np.nan, .1, 0.], [20., np.inf, .1, 0.]])
def test_invalid_energy_grid(grid):
    with pytest.raises(ValueError):
        collapse(grid=grid)


@pytest.mark.parametrize('flux', [[2., -1., 3.], [2., np.nan, 3.], [2., np.inf, 3.],
                                  [0., 1., 2.], [2., 0., 0.], [[2., 3., 5.]], [],
                                  [2., np.finfo(float).max, np.finfo(float).max]])
def test_invalid_flux_or_support(flux):
    with pytest.raises(ValueError):
        collapse(flux=flux)


@pytest.mark.parametrize('defect', ['zero_total_supported', 'bad_vector_shape', 'bad_matrix_shape',
                                    'nan_vector', 'inf_signed_matrix', 'errors', 'other_field',
                                    'gap_order', 'order8', 'unpaired_order', 'overflow_power',
                                    'overflow_chi', 'overflow_loss'])
def test_invalid_native_means_and_derived_overflows(defect):
    data = fields()
    if defect == 'zero_total_supported':
        data['INF_TOT'][0] = 0
    elif defect == 'bad_vector_shape':
        data['INF_TOT'] = np.ones((3, 1))
    elif defect == 'bad_matrix_shape':
        data['INF_S1'] = np.ones(9)
    elif defect == 'nan_vector':
        data['INF_RABSXS'][0] = np.nan
    elif defect == 'inf_signed_matrix':
        data['INF_SP2'][0, 0] = np.inf
    elif defect == 'errors':
        data['INF_TOT'] = np.ones((3, 2))
    elif defect == 'other_field':
        data['INF_FLX'] = np.ones(3)
    elif defect == 'gap_order':
        del data['INF_S1'], data['INF_SP1']
    elif defect == 'order8':
        data['INF_S8'] = data['INF_SP8'] = np.ones((3, 3))
    elif defect == 'unpaired_order':
        data['INF_S4'] = np.ones((3, 3))
    elif defect == 'overflow_power':
        data['INF_FISS'][:] = data['INF_KAPPA'][:] = 1e200
    elif defect == 'overflow_chi':
        data['INF_CHIT'][1:] = np.finfo(float).max
    else:
        data['INF_SP1'][0, 0] = -np.finfo(float).max
        data['INF_TOT'][0] = np.finfo(float).max/4
    with pytest.raises(ValueError):
        collapse(data)


def test_order_seven_ascending_grid_and_optional_absence():
    data = fields(3, 8)
    del data['INF_ABS']
    result = collapse(data, grid=[0., .1, 1., 20.])
    assert tuple(result.collision) == tuple(range(8))
    assert 'INF_ABS' not in result.raw
    np.testing.assert_array_equal(result.provenance['energy_bounds_mev'], [0., .1, 20.])
