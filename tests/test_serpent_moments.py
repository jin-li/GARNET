"""Synthetic signed native moment data only; no external XS or solver runs."""
import numpy as np
import pytest
from ndgpu.serpent_xs import read_serpent_moment_xs, read_serpent_xs


def write(tmp_path, overrides=None, extra=''):
    fields=dict(VERSION="'Serpent 2.2.2'",GC_UNIVERSE_NAME="'synthetic'",MACRO_NG='2',MACRO_E='[20 1 0]')
    vectors={'TOT':[1,2],'ABS':[.4,.8],'RABSXS':[.4,.6],'NSF':[.2,.1],'CHIT':[1,0],'FLX':[2,1],
             'FISS':[.1,.05],'KAPPA':[200,190],'DIFFCOEF':[1,.5]}
    for key,value in vectors.items():fields['INF_'+key]='['+' '.join(f'{x} 0' for x in value)+']'
    for order in range(4):
        for kind in ('S','SP'):
            values=([.2,.4,.1,1.3] if kind=='SP' else [.2,.3,.1,1.1]) if order==0 else [0,-.4,.2,-.1]
            fields[f'INF_{kind}{order}']='['+' '.join(f'{x} {(-.2 if order else 0)}' for x in values)+']'
    fields.update(overrides or {})
    lines=[]
    for key,value in fields.items():
        if value is None:continue
        count=len(value[1:-1]) if value.startswith("'") else len(value.strip('[]').split())
        lines.append(f'{key} (idx,[1:{count}]) = {value};')
    p=tmp_path/'synthetic.m';p.write_text('\n'.join(lines)+extra);return p


def test_signed_pairs_and_explicit_orientation(tmp_path):
    r=read_serpent_moment_xs(write(tmp_path))
    e=r.moment_estimate('SP',1)
    np.testing.assert_array_equal(e.mean,[[0,-.4],[.2,-.1]])
    np.testing.assert_array_equal(e.reported_error,np.full((2,2),-.2))
    m=r.to_moment_material(orientation='from-major')
    v=np.array([2.,3.])
    np.testing.assert_allclose(m.loss[1]@v,[1.4,7.1])
    np.testing.assert_allclose(m.kappa_fission,[20,9.5])
    assert m.provenance['version']=='Serpent 2.2.2'
    assert not m.provenance['physical_admission']
    assert not m.provenance['reported_error_sign_interpreted']
    assert not hasattr(e,'absolute_error')
    np.testing.assert_array_equal(m.production[0],[[.2,.4],[.1,1.3]])
    assert not np.array_equal(m.collision[0],m.production[0])
    with pytest.raises(ValueError):m.loss[0].setflags(write=True)
    with pytest.raises(ValueError):m.raw['INF_SP1'][0,0]=5
    with pytest.raises(TypeError):m.provenance['physical_admission']=True
    r.fields['INF_SP1'][0]=88
    assert m.raw['INF_SP1'][0,0]==0


def test_p0_canonical_reduction(tmp_path):
    r=read_serpent_moment_xs(write(tmp_path))
    m=r.to_moment_material(orientation='from-major')
    d=r.to_diffusion_material(diffusion='inf')
    scatter=d.sigma_s.copy();np.fill_diagonal(scatter,0)
    np.testing.assert_allclose(m.loss[0],np.diag(d.removal)-scatter.T)
    np.testing.assert_array_equal(m.chi,d.chi)
    np.testing.assert_allclose(m.balance_diagnostics['total_minus_reduced_absorption_production_out'],0,atol=1e-15)


@pytest.mark.parametrize('value',['[1 0]','[1 0 2 0 3 0 nan 0]','[1 0 2 0 3 0 4 inf]','[ones(8)]'])
def test_malformed_higher_field(tmp_path,value):
    with pytest.raises(ValueError):read_serpent_moment_xs(write(tmp_path,{'INF_SP1':value}))


def test_missing_duplicate_unknown_and_orientation(tmp_path):
    with pytest.raises(ValueError,match='missing'):read_serpent_moment_xs(write(tmp_path,{'INF_S2':None}))
    with pytest.raises(ValueError,match='duplicate'):read_serpent_moment_xs(write(tmp_path,extra='\nINF_SP1(idx,[1:8])=[0 0 0 0 0 0 0 0];'))
    with pytest.raises(ValueError,match='unsupported moment'):read_serpent_moment_xs(write(tmp_path,{'INF_SP8':'[0 0]'}))
    r=read_serpent_moment_xs(write(tmp_path))
    with pytest.raises(ValueError,match='orientation'):r.to_moment_material(orientation='auto')
    with pytest.raises(ValueError,match='consecutive'):read_serpent_moment_xs(write(tmp_path),required_orders=(0,2))
    with pytest.raises(ValueError,match='INF_FISS'):read_serpent_moment_xs(write(tmp_path,{'INF_FISS':None,'INF_KAPPA':None})).to_moment_material(orientation='from-major')


def test_legacy_higher_fields_still_ignored_without_global_mode(tmp_path):
    p=write(tmp_path,{'INF_SP1':'[unrecognized_expression()]'})
    r=read_serpent_xs(p)
    assert 'INF_SP1' not in r.fields
    r.to_diffusion_material(diffusion='inf')
    with pytest.raises(ValueError):read_serpent_moment_xs(p)
    assert read_serpent_xs(p).n_groups==2


def test_axis_contract_and_group_permutation(tmp_path):
    r=read_serpent_moment_xs(write(tmp_path));a=r.to_moment_material(orientation='from-major')
    for order in range(4):
        for kind in ('S','SP'):
            key=f'INF_{kind}{order}';pairs=r.fields[key].reshape(2,2,2)
            r.fields[key]=pairs.transpose(1,0,2).ravel().copy()
    b=r.to_moment_material(orientation='to-major')
    for order in range(4):np.testing.assert_array_equal(a.loss[order],b.loss[order])
    for key,value in list(r.fields.items()):
        if isinstance(value,np.ndarray) and key.startswith('INF_'):
            if value.size==8:r.fields[key]=value.reshape(2,2,2)[::-1,::-1,:].ravel().copy()
            elif value.size==4:r.fields[key]=value.reshape(2,2)[::-1,:].ravel().copy()
    c=r.to_moment_material(orientation='to-major')
    for order in range(4):np.testing.assert_array_equal(c.loss[order],b.loss[order][::-1,::-1])


def test_optional_order_seven_and_version_not_general_validation(tmp_path):
    fields={'VERSION':"'Serpent future-version unverified'"}
    for order in range(4,8):
        fields[f'INF_S{order}']='[0 -0.1 -0.2 -0.3 0 -0.4 0 -0.5]'
        fields[f'INF_SP{order}']='[0 -0.1 -0.2 -0.3 0 -0.4 0 -0.5]'
    r=read_serpent_moment_xs(write(tmp_path,fields),required_orders=tuple(range(8)))
    m=r.to_moment_material(orientation='from-major')
    assert tuple(m.loss)==tuple(range(8))
    assert m.raw['INF_SP7'][0,1]==-.2
    assert m.reported_errors['INF_SP7'][0,1]==-.3
    assert m.provenance['version']=='Serpent future-version unverified'
    assert m.provenance['local_availability_not_general_version_support']
    with pytest.raises(ValueError,match='VERSION'):
        read_serpent_moment_xs(write(tmp_path,{'VERSION':'2.2'}))


def test_unknown_higher_assignment_syntax_and_optional_malformed(tmp_path):
    with pytest.raises(ValueError,match='unsupported field assignment'):
        read_serpent_moment_xs(write(tmp_path,extra='\nINF_SP3(idx,unknown)=[0];'))
    with pytest.raises(ValueError,match='INF_SP7'):
        read_serpent_moment_xs(write(tmp_path,{'INF_SP7':'[1 0]'}))
