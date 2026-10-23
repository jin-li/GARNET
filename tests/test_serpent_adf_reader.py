"""Synthetic native ADF reader checks; real probes are optional local inputs."""
import os
from pathlib import Path

import numpy as np
import pytest

from garnet.serpent_xs import read_serpent_results, read_serpent_xs


def paired(means, errors=None):
    means = np.asarray(means).ravel()
    errors = np.full(means.shape,.03) if errors is None else np.asarray(errors).ravel()
    return np.column_stack((means,errors)).ravel()


def fields(nfaces=6):
    count = nfaces*2
    density = np.arange(1.,count+1.)
    return {
        "VERSION": "Serpent 2.2.2", "GC_UNIVERSE_NAME": "synthetic-cuboid",
        "MACRO_NG": np.array([2]), "MACRO_E": np.array([20.,.625e-6,0.]),
        "INF_TOT": paired([.5,1.]), "INF_ABS": paired([.01,.1]),
        "INF_RABSXS": paired([.01,.1]), "INF_NSF": paired([.005,.15]),
        "INF_CHIT": paired([.95,.05]), "INF_FLX": paired([3.,1.]),
        "INF_S0": paired([.44,.05,.001,.899]),
        "INF_SP0": paired([.44,.05,.001,.899]),
        "CMM_DIFFCOEF": paired([1.25,.25]),
        "DF_N_SURF": np.array([nfaces]), "DF_VOLUME": np.array([60.]),
        "DF_SURF_AREA": np.arange(10.,10.+nfaces),
        "DF_HET_VOL_FLUX": paired([3.,1.]),
        "DF_HOM_VOL_FLUX": paired([2.9,1.1]),
        "DF_HET_SURF_FLUX": paired(density,np.arange(count)/100),
        "DF_HOM_SURF_FLUX": paired(density+.5),
        "DF_SURF_DF": paired(density/(density+.5),np.zeros(count)),
        "DF_SURF_IN_CURR": paired(np.full(count,30.)),
        "DF_SURF_OUT_CURR": paired(30.-np.arange(-count/2,count/2)),
        "DF_SURF_NET_CURR": paired(np.arange(-count/2,count/2),np.arange(count)/20),
    }


def write(tmp_path, values=None):
    values = fields() if values is None else values
    lines = []
    for key,value in values.items():
        if isinstance(value,str):
            literal = f"'{value}'"
            length = len(value)
        else:
            value = np.asarray(value).ravel()
            length = len(value)
            literal = "[ " + " ".join(format(v,".15g") for v in value) + " ]"
        lines.append(f"{key} (idx, [1:{length}]) = {literal};")
    path = tmp_path/"synthetic_adf_res.m"
    path.write_text("\n".join(lines)+"\n")
    return path


@pytest.mark.parametrize("nfaces",[4,6])
def test_face_major_group_inner_signed_net_and_native_geometry(tmp_path,nfaces):
    raw = fields(nfaces)
    record = read_serpent_xs(write(tmp_path,raw))
    assert record.version == "Serpent 2.2.2"
    assert record.n_surfaces == nfaces
    np.testing.assert_array_equal(record.fields["DF_VOLUME"],[60.])
    np.testing.assert_array_equal(record.fields["DF_SURF_AREA"],np.arange(10.,10.+nfaces))
    density = record.adf_estimate("DF_HET_SURF_FLUX")
    np.testing.assert_array_equal(density.mean,np.arange(1.,2*nfaces+1.).reshape(nfaces,2))
    np.testing.assert_array_equal(density.relative_error,np.arange(2*nfaces).reshape(nfaces,2)/100)
    # Native surface averages and integrated currents are never area-rescaled.
    net = record.adf_estimate("DF_SURF_NET_CURR")
    np.testing.assert_array_equal(net.mean,np.arange(-nfaces,nfaces).reshape(nfaces,2))
    np.testing.assert_array_equal(net.relative_error,np.arange(2*nfaces).reshape(nfaces,2)/20)
    np.testing.assert_allclose(net.absolute_error,np.abs(net.mean)*net.relative_error)
    inbound = record.adf_estimate("DF_SURF_IN_CURR").mean
    outbound = record.adf_estimate("DF_SURF_OUT_CURR").mean
    np.testing.assert_array_equal(net.mean,inbound-outbound)
    np.testing.assert_array_equal(record.adf_estimate("DF_HET_VOL_FLUX").mean,[3.,1.])
    assert record.adf_estimate("DF_HOM_VOL_FLUX").mean.shape == (2,)
    np.testing.assert_array_equal(record.adf_estimate("DF_SURF_DF").relative_error,0)
    # Returned estimates are copies; changing them cannot alter raw evidence.
    density.mean[:] = -1
    assert np.all(record.adf_estimate("DF_HET_SURF_FLUX").mean > 0)


@pytest.mark.parametrize("key,value",[
    ("DF_N_SURF",[0]),("DF_N_SURF",[6.5]),("DF_N_SURF",[np.nan]),
    ("DF_N_SURF",[np.inf]),("DF_N_SURF",[6,6]),
    ("DF_VOLUME",[0]),("DF_VOLUME",[-60]),("DF_VOLUME",[np.inf]),
    ("DF_VOLUME",[60,60]),("DF_SURF_AREA",[10]*5),
    ("DF_SURF_AREA",[0]*6),("DF_SURF_AREA",[np.nan]*6),
    ("DF_HET_VOL_FLUX",paired([3.])),
    ("DF_SURF_NET_CURR",paired([1]*11)),
    ("DF_HET_SURF_FLUX",paired([-1]*12)),
    ("DF_SURF_NET_CURR",paired([np.inf]*12)),
    ("DF_SURF_NET_CURR",paired([1]*12,[-.1]*12)),
    ("DF_SURF_DF",paired([1]*12,[np.nan]*12)),
    ("DF_HOM_SURF_FLUX",np.ones(23)),
])
def test_invalid_geometry_statistics_and_lengths(tmp_path,key,value):
    raw = fields() | {key:np.asarray(value)}
    with pytest.raises(ValueError,match=key):
        read_serpent_xs(write(tmp_path,raw))


def test_missing_surface_count_and_invalid_declared_length(tmp_path):
    raw = fields()
    raw.pop("DF_N_SURF")
    with pytest.raises(ValueError,match="DF_N_SURF"):
        read_serpent_xs(write(tmp_path,raw))
    path = write(tmp_path)
    path.write_text(path.read_text().replace("DF_SURF_NET_CURR (idx, [1:24])","DF_SURF_NET_CURR (idx, [1:22])"))
    with pytest.raises(ValueError,match="DF_SURF_NET_CURR.*declared length"):
        read_serpent_xs(path)


def test_adf_fields_are_protected_from_expressions_and_bad_indexing(tmp_path):
    path = write(tmp_path)
    text = path.read_text()
    for statement in ("DF_VOLUME (idx, 1) = system('false');",
                      "DF_SURF_DF (idx, [2:25]) = [1 0];"):
        path.write_text(text + statement)
        with pytest.raises(ValueError,match="DF_"):
            read_serpent_xs(path)


def test_non_adf_field_and_missing_estimate_rejected(tmp_path):
    raw = fields()
    raw.pop("DF_HOM_VOL_FLUX")
    record = read_serpent_xs(write(tmp_path,raw))
    with pytest.raises(ValueError,match="not a recognized ADF"):
        record.adf_estimate("INF_FLX")
    with pytest.raises(ValueError,match="missing field"):
        record.adf_estimate("DF_HOM_VOL_FLUX")


def test_optional_native_six_face_core_probe():
    source = os.environ.get("GARNET_SERPENT_ADF_CORE_PROBE")
    if not source:
        pytest.skip("set GARNET_SERPENT_ADF_CORE_PROBE to an existing local result file")
    path = Path(source)
    assert path.is_file()
    records = read_serpent_results(path)
    adf = [record for record in records if "DF_N_SURF" in record.fields]
    assert adf
    assert all(record.n_surfaces == 6 and record.n_groups == 4 for record in adf)
    for record in adf:
        for key in ("DF_HET_SURF_FLUX","DF_HOM_SURF_FLUX","DF_SURF_DF",
                    "DF_SURF_NET_CURR","DF_SURF_IN_CURR","DF_SURF_OUT_CURR"):
            est = record.adf_estimate(key)
            assert est.mean.shape == est.relative_error.shape == (6,4)
        inbound = record.adf_estimate("DF_SURF_IN_CURR").mean
        outbound = record.adf_estimate("DF_SURF_OUT_CURR").mean
        net = record.adf_estimate("DF_SURF_NET_CURR").mean
        # Account for independent rounding of printed, almost-cancelling tallies.
        np.testing.assert_allclose(net,inbound-outbound,rtol=0,atol=2e-5*np.max(inbound+outbound))
    assert any(np.any(record.adf_estimate("DF_SURF_NET_CURR").mean < 0) for record in adf)


def test_optional_unqualified_core_probe_preserves_strict_xs_gate():
    source = os.environ.get("GARNET_SERPENT_ADF_INVALID_CORE_PROBE")
    if not source:
        pytest.skip("set GARNET_SERPENT_ADF_INVALID_CORE_PROBE to an unqualified local result file")
    assert Path(source).is_file()
    with pytest.raises(ValueError,match="INF_(?:DIFFCOEF|TOT): values must be strictly positive"):
        read_serpent_results(source)


def test_explicit_trc_preserves_native_values_and_default_cmm(tmp_path):
    raw=fields();raw['TRC_DIFFCOEF']=paired([1.3,.26],[.02,.04])
    record=read_serpent_xs(write(tmp_path,raw))
    np.testing.assert_allclose(record.to_diffusion_material(diffusion='trc').diffusion,[1.3,.26])
    np.testing.assert_allclose(record.to_diffusion_material().diffusion,[1.25,.25])
    np.testing.assert_array_equal(record.fields['TRC_DIFFCOEF'],raw['TRC_DIFFCOEF'])
    np.testing.assert_allclose(record.estimate('TRC_DIFFCOEF').relative_error,[.02,.04])


def test_missing_trc_rejects_without_cmm_fallback(tmp_path):
    record=read_serpent_xs(write(tmp_path))
    with pytest.raises(ValueError,match='TRC_DIFFCOEF.*missing'):
        record.to_diffusion_material(diffusion='trc')


@pytest.mark.parametrize('means', [[0,.2],[-1,.2],[float('nan'),.2]])
def test_invalid_native_trc_rejects(tmp_path,means):
    raw=fields();raw['TRC_DIFFCOEF']=paired(means)
    with pytest.raises(ValueError,match='TRC_DIFFCOEF'):
        read_serpent_xs(write(tmp_path,raw))
