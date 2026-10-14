"""Raw signed current-moment decoding only; no sign-flux interpretation."""
import numpy as np
import pytest

from ndgpu.serpent_xs import read_serpent_results

FIELDS = ('DF_SGN_SURF_IN_CURR', 'DF_SGN_SURF_OUT_CURR', 'DF_SGN_SURF_NET_CURR')


def paired(means, errors=None):
    values = np.asarray(means).ravel()
    error = np.full(values.size, .02) if errors is None else np.asarray(errors).ravel()
    return np.column_stack((values, error)).ravel()


def base():
    return dict(VERSION='Serpent 2.2.2', GC_UNIVERSE_NAME='synthetic', MACRO_NG=[2], MACRO_E=[20, 1e-6, 0],
                INF_TOT=paired([1, 1]), INF_ABS=paired([.1, .2]), INF_RABSXS=paired([.1, .2]),
                INF_NSF=paired([0, 0]), INF_FLX=paired([1, 2]), INF_S0=paired([0, 0, 0, 0]),
                INF_SP0=paired([0, 0, 0, 0]))


def write(tmp_path, values):
    lines = []
    for key, value in values.items():
        if isinstance(value, str):
            literal, count = repr(value), len(value)
        else:
            value = np.asarray(value).ravel()
            literal, count = '[ '+' '.join(format(v, '.16g') for v in value)+' ]', value.size
        lines.append(f'{key} (idx, [1:{count}]) = {literal};')
    path = tmp_path/'signed_res.m'
    path.write_text('\n'.join(lines)+'\n')
    return path


def signed(ns=3):
    values = base() | dict(DF_N_SGN=[ns])
    for index, field in enumerate(FIELDS):
        values[field] = paired(np.arange(2*ns).reshape(ns, 2)-4.5-index)
    return values


def test_negative_in_out_net_raw_moments_group_fastest_without_face_assumption(tmp_path):
    values = signed()
    record, = read_serpent_results(write(tmp_path, values))
    assert record.n_sign_moments == 3  # deliberately not cuboid12, no DF_N_SURF required
    for index, field in enumerate(FIELDS):
        result = record.adf_sign_current_estimate(field)
        np.testing.assert_array_equal(result.mean, np.arange(6).reshape(3, 2)-4.5-index)
        np.testing.assert_array_equal(result.relative_error, np.full((3, 2), .02))
        np.testing.assert_allclose(result.absolute_error, abs(result.mean)*.02)
        result.mean[:] = 99
        assert np.any(record.adf_sign_current_estimate(field).mean < 0)  # copy, no mutation


def test_optional_absent_or_zero_metadata_preserves_ordinary_records(tmp_path):
    for additional in ({}, dict(DF_N_SGN=[0])):
        record, = read_serpent_results(write(tmp_path, base() | additional))
        np.testing.assert_array_equal(record.estimate('INF_FLX').mean, [1, 2])
        if additional:
            assert record.n_sign_moments == 0
        else:
            with pytest.raises(ValueError, match='DF_N_SGN'):
                _ = record.n_sign_moments
        with pytest.raises(ValueError, match='missing field'):
            record.adf_sign_current_estimate(FIELDS[0])


def test_zero_count_empty_optional_sign_current_arrays(tmp_path):
    record, = read_serpent_results(write(tmp_path, signed(0)))
    assert record.n_sign_moments == 0
    for field in FIELDS:
        assert record.adf_sign_current_estimate(field).mean.shape == (0, 2)


@pytest.mark.parametrize('metadata', [[-1], [1.5], [np.inf], [np.nan], [2, 3]])
def test_invalid_sign_count_rejected(tmp_path, metadata):
    with pytest.raises(ValueError, match='DF_N_SGN'):
        read_serpent_results(write(tmp_path, signed() | dict(DF_N_SGN=metadata)))


def test_missing_sign_count_and_shape_mismatch_rejected(tmp_path):
    values = signed()
    del values['DF_N_SGN']
    with pytest.raises(ValueError, match='DF_N_SGN'):
        read_serpent_results(write(tmp_path, values))
    with pytest.raises(ValueError, match='expected 6 mean/error pairs'):
        read_serpent_results(write(tmp_path, signed() | {FIELDS[0]: paired([1, 2])}))


@pytest.mark.parametrize('field,index,value', [(FIELDS[0], 0, np.nan), (FIELDS[1], 0, np.inf),
    (FIELDS[2], 1, -.01), (FIELDS[0], 1, np.inf), (FIELDS[1], 1, np.nan)])
def test_invalid_signed_means_or_errors_rejected(tmp_path, field, index, value):
    values = signed()
    values[field][index] = value
    with pytest.raises(ValueError, match='finite'):
        read_serpent_results(write(tmp_path, values))


def test_new_recognized_fields_cannot_hide_nonliteral_scores(tmp_path):
    path = write(tmp_path, signed())
    path.write_text(path.read_text().replace('DF_N_SGN (idx, [1:1]) = [ 3 ];',
                                             'DF_N_SGN (idx, [1:1]) = unsafe_expression();'))
    with pytest.raises(ValueError, match='DF_N_SGN'):
        read_serpent_results(path)


def test_sign_flux_fields_not_decoded_and_ordinary_incoming_current_stays_nonnegative(tmp_path):
    values = signed() | dict(DF_SGN_HET_FLUX=paired([-1, 2]), DF_SGN_HOM_FLUX=paired([3, -4]))
    record, = read_serpent_results(write(tmp_path, values))
    for name in ('DF_SGN_HET_FLUX', 'DF_SGN_HOM_FLUX', 'DF_HET_SURF_FLUX'):
        with pytest.raises(ValueError, match='not a recognized sign-current'):
            record.adf_sign_current_estimate(name)
    for name in ('DF_SGN_HET_FLUX', 'DF_SGN_HOM_FLUX'):
        with pytest.raises(ValueError, match='not a recognized'):
            record.estimate(name)
    bad = base() | dict(DF_N_SURF=[1], DF_SURF_IN_CURR=paired([-1, 2]))
    with pytest.raises(ValueError, match='nonnegative'):
        read_serpent_results(write(tmp_path, bad))
