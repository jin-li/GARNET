"""Independent scalar-rate checks; no transport or uncertainty qualification."""
from dataclasses import replace

import numpy as np
import pytest

from garnet.moment_homogenization import homogenize_moment_materials
from garnet.moment_materials import MomentMaterial


def material(index=0, **changes):
    total = np.array([2., 3.])+index
    nsf = np.array([.4, .1]) if index == 0 else np.array([.1, .9])
    fiss = np.array([.2, .05]) if index == 0 else np.array([.04, .3])
    kappa = np.array([190., 210.]) if index == 0 else np.array([230., 180.])
    chi = np.array([.9, .3]) if index == 0 else np.array([.2, .9])
    collision = {o: np.array([[.3, .12], [.04, .2]])*(index+1)/(o+1)
                 for o in range(4)}
    production = {o: np.array([[.4, .17], [.06, .25]])*(index+1)/(o+1)
                  for o in range(4)}
    for o in (1, 2, 3):
        collision[o][0, 1] *= -1
        production[o][1, 0] *= -1
    raw = dict(INF_TOT=total, INF_NSF=nsf, INF_RABSXS=np.array([.8, .9])+index,
               INF_FISS=fiss, INF_KAPPA=kappa, INF_CHIT=chi)
    raw.update({f'INF_S{o}': a for o, a in collision.items()})
    raw.update({f'INF_SP{o}': a for o, a in production.items()})
    result = MomentMaterial(total, chi, nsf, fiss*kappa, collision, production,
                            {o: np.diag(total)-a.T for o, a in production.items()},
                            raw, {'INF_TOT': np.array([.01, .02])},
                            dict(orientation_contract='from-major', orders=(0, 1, 2, 3),
                                 energy_bounds_mev=np.array([20., .1, 0.]),
                                 source=f'source-{index}', universe=f'ring-{index}',
                                 source_sha256=f'hash-{index}', physical_admission=False), {})
    return replace(result, **changes)


def fixture():
    # Unequal volumes and group-dependent densities yield already integrated
    # weights. Only this product, never volumes a second time, enters the API.
    volumes = np.array([2., 5.])
    density = np.array([[1., 4.], [3., .6]])
    return (material(), material(1)), volumes[:, None]*density, volumes


def test_unequal_integrated_weights_preserve_all_scalar_numerators_and_losses():
    inputs, flux, volumes = fixture()
    pooled = homogenize_moment_materials(inputs, flux)
    support = flux.sum(axis=0)
    for attr in ('total', 'nu_sigma_f', 'kappa_fission'):
        numerator = sum(flux[i]*getattr(m, attr) for i, m in enumerate(inputs))
        np.testing.assert_allclose(getattr(pooled, attr)*support, numerator, rtol=2e-15)
    for field in ('INF_RABSXS', 'INF_FISS'):
        np.testing.assert_allclose(pooled.raw[field]*support,
                                   sum(flux[i]*m.raw[field] for i, m in enumerate(inputs)))
    wrong = sum(flux[i]*volumes[i]*m.total for i, m in enumerate(inputs)) / (flux*volumes[:, None]).sum(axis=0)
    assert not np.allclose(pooled.total, wrong)
    assert np.linalg.norm(inputs[0].loss[1]@inputs[0].loss[2]
                          -inputs[0].loss[2]@inputs[0].loss[1]) > 1e-4
    for order in range(4):
        for attr in ('collision', 'production'):
            expected = sum(flux[i][:, None]*getattr(m, attr)[order] for i, m in enumerate(inputs))
            np.testing.assert_allclose(getattr(pooled, attr)[order]*support[:, None], expected)
        np.testing.assert_allclose(pooled.loss[order], np.diag(pooled.total)-pooled.production[order].T)
        np.testing.assert_allclose(pooled.loss[order]@support,
                                   sum(m.loss[order]@flux[i] for i, m in enumerate(inputs)))
    assert pooled.production[3][1, 0] < 0
    assert pooled.collision[2][0, 1] < 0
    expected_residual = sum(flux[i]*(m.total-m.raw['INF_RABSXS']-m.production[0].sum(axis=1))
                            for i, m in enumerate(inputs))/support
    np.testing.assert_allclose(pooled.balance_diagnostics['total_minus_reduced_absorption_production_out'],
                               expected_residual)


def test_source_weighted_chi_preserves_each_emitted_group_without_normalization():
    inputs, flux, _ = fixture()
    pooled = homogenize_moment_materials(inputs, flux)
    sources = [m.nu_sigma_f@flux[i] for i, m in enumerate(inputs)]
    np.testing.assert_allclose(pooled.chi*(pooled.nu_sigma_f@flux.sum(axis=0)),
                               sum(source*m.chi for source, m in zip(sources, inputs)))
    assert pooled.chi.sum() != pytest.approx(1.)
    group_weighted_wrong = sum(flux[i]*m.chi for i, m in enumerate(inputs))/flux.sum(axis=0)
    assert not np.allclose(pooled.chi, group_weighted_wrong)


def test_power_uses_direct_product_pool_and_changes_with_new_fv_flux():
    inputs, flux, _ = fixture()
    pooled = homogenize_moment_materials(inputs, flux)
    separate_kappa = sum(flux[i]*m.raw['INF_KAPPA'] for i, m in enumerate(inputs))/flux.sum(axis=0)
    assert not np.allclose(pooled.kappa_fission, pooled.raw['INF_FISS']*separate_kappa)
    np.testing.assert_allclose(pooled.raw['INF_FISS']*pooled.raw['INF_KAPPA'], pooled.kappa_fission)
    # A later FV shape must use the pooled material, not the original ring XS.
    fv_flux = np.array([[5., 1.], [1., 7.]])
    pooled_power = sum(pooled.kappa_fission@row for row in fv_flux)
    original_power = sum(m.kappa_fission@fv_flux[i] for i, m in enumerate(inputs))
    separate_power = (pooled.raw['INF_FISS']*separate_kappa)@fv_flux.sum(axis=0)
    assert pooled_power != pytest.approx(original_power)
    assert pooled_power != pytest.approx(separate_power)


def test_singleton_exact_identity_and_aggregate_immutability_provenance():
    inputs, flux, _ = fixture()
    assert homogenize_moment_materials([inputs[0]], flux[:1]) is inputs[0]
    pooled = homogenize_moment_materials(inputs, flux)
    assert pooled.reported_errors == {}
    assert pooled.provenance['aggregate_uncertainty'] == 'unknown'
    assert pooled.provenance['derived_coefficient_means']
    assert not pooled.provenance['raw_fields_are_native_estimates']
    assert not pooled.provenance['physical_admission']
    assert pooled.provenance['source_provenance'][1]['universe'] == 'ring-1'
    assert pooled.provenance['source_provenance'][0]['source_sha256'] == 'hash-0'
    for array in (pooled.total, pooled.raw['INF_FISS'], pooled.production[2],
                  pooled.provenance['volume_integrated_scalar_flux']):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        pooled.provenance['source_provenance'][0]['source'] = 'changed'
    np.testing.assert_array_equal(inputs[0].total, [2., 3.])


@pytest.mark.parametrize('flux', [np.ones(2), np.ones((2, 3)), [[1., -1.], [2., 3.]],
                                  [[1., np.nan], [2., 3.]], [[1., np.inf], [2., 3.]],
                                  [[0., 1.], [0., 2.]], np.zeros((2, 2))])
def test_invalid_weights_refused(flux):
    with pytest.raises(ValueError):
        homogenize_moment_materials([material(), material(1)], flux)


@pytest.mark.parametrize('defect', ['grid', 'missing_grid', 'orientation', 'order', 'missing_raw',
                                    'nonfinite', 'shape', 'loss', 'power'])
def test_incompatible_or_incomplete_native_materials_refused(defect):
    first, second = material(), material(1)
    if defect in ('grid', 'missing_grid', 'orientation', 'order'):
        p = dict(second.provenance)
        if defect == 'grid':
            p['energy_bounds_mev'] = np.array([20., .2, 0.])
        elif defect == 'missing_grid':
            del p['energy_bounds_mev']
        elif defect == 'orientation':
            p['orientation_contract'] = 'to-major'
        else:
            p['orders'] = (0, 1)
        second = replace(second, provenance=p)
    elif defect == 'missing_raw':
        raw = dict(second.raw); del raw['INF_RABSXS']
        second = replace(second, raw=raw)
    elif defect == 'nonfinite':
        second = replace(second, total=np.array([np.nan, 3.]))
    elif defect == 'shape':
        second = replace(second, collision={**second.collision, 2: np.ones((1, 2))})
    elif defect == 'loss':
        second = replace(second, loss={**second.loss, 1: second.loss[1].T})
    else:
        second = replace(second, kappa_fission=second.kappa_fission*2)
    with pytest.raises(ValueError):
        homogenize_moment_materials([first, second], np.ones((2, 2)))


def test_zero_source_refused_even_for_singleton_and_missing_material_refused():
    m = material()
    raw = {**m.raw, 'INF_NSF': np.zeros(2)}
    nonfissile = replace(m, nu_sigma_f=np.zeros(2), raw=raw)
    with pytest.raises(ValueError, match='fission source'):
        homogenize_moment_materials([nonfissile], [[1., 1.]])
    for inputs in ([], [m, None]):
        with pytest.raises(ValueError):
            homogenize_moment_materials(inputs, np.ones((len(inputs), 2)))


def test_zero_fission_group_keeps_direct_power_and_omits_undefined_raw_kappa():
    inputs = []
    for m in (material(), material(1)):
        fiss = np.array([m.raw['INF_FISS'][0], 0.])
        inputs.append(replace(m, raw={**m.raw, 'INF_FISS': fiss},
                              kappa_fission=fiss*m.raw['INF_KAPPA']))
    pooled = homogenize_moment_materials(inputs, np.ones((2, 2)))
    assert 'INF_KAPPA' not in pooled.raw
    assert pooled.kappa_fission[1] == 0.
    assert not pooled.provenance['raw_kappa_defined']
    # The explicitly undefined raw ratio also remains safe in a later pool.
    nested = homogenize_moment_materials([pooled, pooled], np.ones((2, 2)))
    np.testing.assert_array_equal(nested.kappa_fission, pooled.kappa_fission)


def test_unsupported_power_and_overflow_refused_without_fallback():
    m = material()
    raw = {**m.raw, 'INF_FISS': np.zeros(2)}
    del raw['INF_KAPPA']
    with pytest.raises(ValueError, match='zero fission'):
        homogenize_moment_materials([replace(m, raw=raw)], [[1., 1.]])
    with pytest.raises(ValueError, match='finite flux support'):
        homogenize_moment_materials([m, m], np.full((2, 2), np.finfo(float).max))


def test_signed_reduced_absorption_is_preserved_without_balance_repair():
    inputs, flux, _ = fixture()
    inputs = [replace(m, raw={**m.raw, 'INF_RABSXS': -m.raw['INF_RABSXS']})
              for m in inputs]
    pooled = homogenize_moment_materials(inputs, flux)
    expected = sum(flux[i]*m.raw['INF_RABSXS'] for i, m in enumerate(inputs))
    np.testing.assert_allclose(pooled.raw['INF_RABSXS']*flux.sum(axis=0), expected)
    assert np.all(pooled.raw['INF_RABSXS'] < 0)


@pytest.mark.parametrize('kind', ['collision', 'production'])
def test_negative_zero_order_refused_while_signed_higher_orders_are_supported(kind):
    m = material()
    mapping = dict(getattr(m, kind))
    mapping[0] = -mapping[0]
    with pytest.raises(ValueError, match='zero-order'):
        homogenize_moment_materials([replace(m, **{kind: mapping})], [[1., 1.]])
