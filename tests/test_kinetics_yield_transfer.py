"""Raw yield transfer is explicit modeling and preserves column composition."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ndgpu.kinetics_yield_transfer import transfer_delayed_yields


KW = dict(source_domain='physical D4 orbit',receiver_domain='representative virtual GCU',
          qualification='Caller independently qualified homologous regions, groups and family definitions.')


def example():
    source = np.array([[2.,4.],[5.,10.]])
    delayed = np.array([[[[.01,.02],[.03,.04]],[[.05,.06],[.07,.08]]],
                        [[[.1,.2],[.3,.4]],[[.5,.6],[.7,.8]]]])
    receiver = np.array([[6.,2.],[5.,20.]])
    return source,delayed,receiver


def test_manual_column_scaling_and_channel_spectra_preserved():
    source,delayed,receiver = example()
    result = transfer_delayed_yields(source,delayed,receiver,**KW)
    expected = np.array([[[[.03,.01],[.09,.02]],[[.15,.03],[.21,.04]]],
                         [[[.1,.4],[.3,.8]],[[.5,1.2],[.7,1.6]]]])
    np.testing.assert_allclose(result.delayed_production,expected,rtol=1e-15)
    # Preserve joint family/emitted-energy probabilities for each incident group.
    for n in range(2):
        for g in range(2):
            np.testing.assert_allclose(result.delayed_production[n,:,:,g]/receiver[n,g],
                                       delayed[n,:,:,g]/source[n,g],rtol=1e-15)
    np.testing.assert_array_equal(result.total_yield_difference,[[4.,-2.],[0.,10.]])
    np.testing.assert_array_equal(result.relative_total_yield_difference,[[2.,-.5],[0.,1.]])
    assert result.max_abs_total_yield_difference == 10
    assert result.max_abs_relative_total_yield_difference == 2
    assert result.source_domain == KW['source_domain']
    assert result.receiver_domain == KW['receiver_domain']
    assert result.qualification == KW['qualification']


def test_zero_groups_channels_and_receiver_yield():
    source = [[0.,4.,2.]]
    receiver = [[0.,2.,0.]]
    delayed = np.zeros((1,2,3,3))
    delayed[0,0,1,1] = .2
    delayed[0,1,2,2] = .1
    result = transfer_delayed_yields(source,delayed,receiver,**KW)
    expected = np.zeros_like(delayed)
    expected[0,0,1,1] = .1
    np.testing.assert_array_equal(result.delayed_production,expected)
    np.testing.assert_array_equal(result.relative_total_yield_difference,[[0.,-.5,-1.]])


def test_zero_source_positive_receiver_rejected_even_with_no_delayed_neutrons():
    with pytest.raises(ValueError,match='zero source denominator'):
        transfer_delayed_yields([[0.]],np.zeros((1,1,1,1)),[[1.]],**KW)


def test_inputs_and_result_are_independent_immutable_copies():
    source,delayed,receiver = example()
    originals = [v.copy() for v in (source,delayed,receiver)]
    result = transfer_delayed_yields(source,delayed,receiver,**KW)
    source[:]=42;delayed[:]=42;receiver[:]=42
    for value,expected in zip((result.source_total_yields,result.source_delayed_production,
                               result.receiver_total_yields),originals):
        np.testing.assert_array_equal(value,expected)
    for name in ('source_total_yields','source_delayed_production','receiver_total_yields',
                 'delayed_production','total_yield_difference','relative_total_yield_difference'):
        value = getattr(result,name)
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flat[0] = 1
    with pytest.raises(FrozenInstanceError):
        result.qualification = 'changed'


@pytest.mark.parametrize('key',['source_domain','receiver_domain','qualification'])
@pytest.mark.parametrize('bad',['',' \n ',None,1,True])
def test_explicit_domains_and_qualification_required(key,bad):
    args = dict(KW);args[key]=bad
    with pytest.raises(ValueError,match=key):
        transfer_delayed_yields(*example(),**args)


@pytest.mark.parametrize('which',[0,1,2])
@pytest.mark.parametrize('badtype',['bool','complex','object','str'])
def test_invalid_array_dtypes_rejected(which,badtype):
    args = list(example())
    args[which] = args[which].astype(badtype)
    with pytest.raises(ValueError,match='real numeric'):
        transfer_delayed_yields(*args,**KW)


@pytest.mark.parametrize('which',[0,1,2])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf,-.01])
def test_invalid_array_values_rejected(which,bad):
    args = list(example());args[which].flat[0] = bad
    with pytest.raises(ValueError):
        transfer_delayed_yields(*args,**KW)


@pytest.mark.parametrize('source,delayed,receiver',[
    ([],np.zeros((0,1,1,1)),[]),
    ([1.],np.zeros((1,1,1,1)),[[1.]]),
    (np.zeros((1,0)),np.zeros((1,1,0,0)),np.zeros((1,0))),
    ([[1.]],np.zeros((1,0,1,1)),[[1.]]),
    ([[1.]],np.zeros((1,1,2,1)),[[1.]]),
    ([[1.]],np.zeros((1,1,1,2)),[[1.]]),
    ([[1.]],np.zeros((2,1,1,1)),[[1.]]),
    ([[1.]],np.zeros((1,1,1,1)),[1.]),
    ([[1.]],np.zeros((1,1,1,1)),[[1.,2.]]),
])
def test_invalid_shapes(source,delayed,receiver):
    with pytest.raises(ValueError):
        transfer_delayed_yields(source,delayed,receiver,**KW)


def test_delayed_column_sum_may_not_exceed_total_even_across_families():
    delayed = np.full((1,2,2,2),.3)
    with pytest.raises(ValueError,match='exceeds source total'):
        transfer_delayed_yields([[1.,1.]],delayed,[[1.,1.]],**KW)
    with pytest.raises(ValueError,match='exceeds source total'):
        transfer_delayed_yields([[0.]],np.array([[[[1e-300]]]]),[[0.]],**KW)


def test_no_mismatch_acceptance_threshold_or_renormalization():
    result = transfer_delayed_yields([[1.]],[[[[.125]]]],[[1e8]],**KW)
    np.testing.assert_array_equal(result.delayed_production,[[[[12500000.]]]])
    assert result.max_abs_relative_total_yield_difference == 99999999


def test_overflowed_column_sum_or_diagnostic_is_explicit_error():
    with pytest.raises(ValueError,match='exceeds source total'):
        transfer_delayed_yields([[1e308]],np.full((1,2,1,1),1e308),[[1e308]],**KW)
    with pytest.raises(ValueError,match='relative total yield difference'):
        transfer_delayed_yields([[1e-308]],np.zeros((1,1,1,1)),[[1e308]],**KW)


@pytest.mark.parametrize('total,delayed',[(1e100,1e-300),(1e308,1e-300),
                                        (1e-100,1e-300),(1.,np.nextafter(0.,1.))])
def test_extreme_equal_yield_roundtrip_preserves_representable_delayed_tail(total,delayed):
    result = transfer_delayed_yields([[total]],[[[[delayed]]]],[[total]],**KW)
    # Equal source/receiver is an identity even when delayed/total underflows.
    assert result.delayed_production[0,0,0,0] == delayed


def test_extreme_scaling_avoids_multiply_first_overflow():
    result = transfer_delayed_yields([[1e308]],[[[[1e307]]]],[[1e308]],**KW)
    assert result.delayed_production[0,0,0,0] == 1e307


def test_positive_unrepresentable_final_coefficient_is_rejected():
    with pytest.raises(ValueError,match='underflows to zero'):
        transfer_delayed_yields([[1e100]],[[[[1e-300]]]],[[1.]],**KW)
    smallest = np.nextafter(0.,1.)
    with pytest.raises(ValueError,match='underflows to zero'):
        transfer_delayed_yields([[1.]],[[[[smallest]]]],[[.25]],**KW)


def test_extreme_zero_receiver_channel_is_allowed_without_underflow_error():
    result = transfer_delayed_yields([[1e100]],[[[[1e-300]]]],[[0.]],**KW)
    assert result.delayed_production[0,0,0,0] == 0


def test_extreme_representable_tail_receives_actual_nonidentity_scaling():
    result = transfer_delayed_yields([[1e100]],[[[[1e-300]]]],[[5e99]],**KW)
    assert result.delayed_production[0,0,0,0] == 5e-301
